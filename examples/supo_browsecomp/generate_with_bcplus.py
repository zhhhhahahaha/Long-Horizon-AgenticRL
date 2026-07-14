"""Multi-turn ReAct rollout + reward for BrowseComp-Plus, following the SUPO
paper's workflow="search" recipe (arXiv 2510.11967).

Four callables are exported and wired into slime via:
    --custom-generate-function-path      examples.supo_browsecomp.generate_with_bcplus.generate
    --custom-rm-path                     examples.supo_browsecomp.generate_with_bcplus.reward_func
    --custom-reward-post-process-path    examples.supo_browsecomp.generate_with_bcplus.reward_post_process
    --custom-rollout-log-function-path   examples.supo_browsecomp.generate_with_bcplus.log_bcplus

log_bcplus is where per-rollout wandb metrics are emitted — it's the only
hook slime passes the driver's rollout_id to, so it's the correct place for
wandb ``rollout/step``-aligned logging (works uniformly for sync and async
training). See notes/rollout_id_data_model.md at the repo root for the full
mental model.

The BrowseComp-Plus parquet carries the trimmed task-role system prompt +
user question. We render the initial prompt via ``apply_chat_template(...,
tools=TOOLS)`` so Qwen3.5's chat template auto-emits the ``<tools>`` schema
block + ``<tool_call>`` format instructions. Each rollout turn calls
SGLang /generate with input_ids, parses the
``<tool_call><function=NAME><parameter=k>v</parameter></function></tool_call>``
XML, executes it against the search server (or terminates on ``finish``),
and appends the observation as a hand-shaped ``<tool_response>`` block plus
the next-turn generation prompt. Loop stops on ``finish`` /
``--rollout-max-turns`` / length budget.

**Context compression (SUPO paper's session-folding mechanism)**: when a
sub-trajectory's total context length (prompt + accumulated response) exceeds
``BCPLUS_COMPRESS_THRESH`` fraction of ``--rollout-max-context-len`` and we
have not hit ``BCPLUS_MAX_SUB_TRAJS``, the current sub-trajectory closes out
with a ``<summary>...</summary>`` block produced by the policy itself (SUPO's
``SUMMARY_PROMPT_SEARCH``), and a fresh sub-trajectory opens with only the
original question + the summary as its starting context. Each sub-trajectory
is a separate training ``Sample``; all siblings from the same rollout
invocation share ``rollout_id`` (matches
``slime/rollout/_fanout_test_helpers.py``) so slime's per-rollout loss
aggregation and DP scheduling group them correctly. The final sibling's
outcome reward is broadcast to all siblings in ``reward_post_process``, which
also implements SUPO's FOLDGRPO grouping (per-prompt ``group_index`` mean/std
with ``rollout_id`` dedup, so a rollout with N sub-trajs still contributes
one reward to the group statistics).

**Compression-failure penalty**: when a compression turn fires but the model
fails to emit a real ``<summary>...</summary>`` block (we salvage raw text
via a fallback path), ``BCPLUS_COMPRESS_PENALTY`` is subtracted from that
one sub-trajectory's score in ``reward_post_process``. Group mean/std are
computed from unpenalized per-rollout final rewards, so the penalty shifts
only the failing sub-traj's advantage — sibling scores are unaffected. This
gives the model a gradient signal to learn to emit real summary blocks
without punishing its siblings when one of them happens to be the failing
one. Track ``bcplus/compress_success_rate`` on wandb to see this improve
over training. Metric ``summary_source`` (per sub-traj) is one of
``"extracted"`` (good), ``"fallback"`` (salvage from raw text, penalized),
or ``"empty"`` (no output at all, penalized).

The reward is a single-model judge routed through the internal MetaGen gateway
(default: gpt-5-4-genai-dss4). SUPO's original two-model fallback path
(gpt-4o-mini primary + gpt-4.1 relaxed-EM fallback) was collapsed because
our primary is already stronger than SUPO's fallback.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import re
import unicodedata
from collections import Counter

import httpx

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .local_search_client import AsyncSearchClient
from .tool_schemas import TOOLS

BCPLUS_CONFIGS = {
    "max_turns": int(os.environ.get("BCPLUS_MAX_TURNS", "64")),
    "search_topk_default": 10,
    "search_topk_cap": 20,
    "doc_words_snippet": 512,  # matches SUPO reference impl (search result snippet)
    "doc_words_full": 4096,    # matches SUPO reference impl (open_page full content)
    "search_concurrency": int(os.environ.get("BCPLUS_SEARCH_CONCURRENCY", "128")),
    "judge_concurrency": int(os.environ.get("BCPLUS_JUDGE_CONCURRENCY", "64")),
    # Judge calls go through the internal MetaGen gateway via the Llama API
    # OpenAI-compatible endpoint (base_url below). Model names are MetaGen ids,
    # not raw OpenAI ids. Requires LLAMA_API_KEY (a "LLM|..." key) entitled to
    # the target model — see README for the entitlement setup.
    "judge_model": os.environ.get("BCPLUS_JUDGE_MODEL", "gpt-5-4-genai-dss4"),
    "judge_base_url": os.environ.get("BCPLUS_JUDGE_BASE_URL", "https://api.llama.com/compat/v1/"),
    "judge_max_retries": 3,
    # Context-compression triggers a new sub-trajectory when total context
    # length (prompt + accumulated response) exceeds this fraction of
    # --rollout-max-context-len. See module docstring above.
    "compress_length_threshold": float(os.environ.get("BCPLUS_COMPRESS_THRESH", "0.85")),
    # Hard cap on sub-trajectories per rollout. Matches SUPO's max_session=5.
    "max_sub_trajs": int(os.environ.get("BCPLUS_MAX_SUB_TRAJS", "5")),
    # Penalty subtracted from a sub-traj's `score` when it was closed by
    # compression but the model failed to emit a real <summary>...</summary>
    # block (summary_source != "extracted"). Applied per-sub-traj in
    # reward_post_process AFTER final-reward broadcast but BEFORE GRPO
    # normalization, so it changes only the failing sub-traj's advantage —
    # sibling advantages and group mean/std are unaffected. Set to 0 to
    # disable the penalty entirely. Shell default in run_qwen3p5_4B.sh
    # RUNTIME_ENV_JSON must match this default.
    "compress_penalty": float(os.environ.get("BCPLUS_COMPRESS_PENALTY", "0.5")),
}

_SEARCH_SEM = asyncio.Semaphore(BCPLUS_CONFIGS["search_concurrency"])
_JUDGE_SEM = asyncio.Semaphore(BCPLUS_CONFIGS["judge_concurrency"])

# One-time flag so we register the bcplus/* -> rollout/step binding once per
# process. Guarded here (not in module import) because wandb.run only exists
# after slime's init_tracking has run inside this Ray actor.
_BCPLUS_METRIC_DEFINED = False

# module-level singleton created lazily on first rollout (needs an event loop)
_SEARCH_CLIENT: AsyncSearchClient | None = None


def _search_client() -> AsyncSearchClient:
    global _SEARCH_CLIENT
    if _SEARCH_CLIENT is None:
        base_url = os.environ["LOCAL_SEARCH_URL"]
        _SEARCH_CLIENT = AsyncSearchClient(base_url=base_url)
    return _SEARCH_CLIENT


# ---------------------------------------------------------------------------
# XML function-call parsing (ported from SUPO reference impl)
# ---------------------------------------------------------------------------


def _extract_fn_call(text: str) -> list[dict] | None:
    if not text:
        return None
    # strip any inline session-tag markers used by SUPO (harmless otherwise)
    text = re.split(r"<\[[^\]]+\]>", text)[-1].strip()
    # Qwen3.5 official format: <tool_call>\n<function=NAME>...\n</function>\n</tool_call>.
    # Sampling stops AT </tool_call> (kept via no_stop_trim=True in GenerateState),
    # so most calls end with the closing tag. Fall back to end-of-string in case
    # the model emitted a truncated tool call.
    matches = list(re.finditer(
        r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*(?:</tool_call>|$)",
        text,
        re.DOTALL,
    ))
    if not matches:
        return None
    # group consecutive calls; take the last group (last block emitted by the model)
    groups: list[list[re.Match]] = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count("\n", prev.end(), m.start())
        if line_gap < 4:
            groups[-1].append(m)
        else:
            groups.append([m])
    last = groups[-1]
    return [
        {
            "function": m.group(1),
            "arguments": {
                # Qwen prefers <parameter=k>\n{v}\n</parameter> but the model
                # doesn't always emit the surrounding newlines. Strip so both
                # forms produce identical arg values.
                name: value.strip()
                for name, value in re.findall(
                    r"<parameter=([^>]+)>(.*?)</parameter>", m.group(2), re.DOTALL
                )
            },
        }
        for m in last
    ]


def _wrap_observation_and_reopen_assistant(obs: str) -> str:
    """Byte-exact reproduction of what Qwen3.5 chat template emits for
    ``{"role": "tool", "content": obs}`` followed by
    ``add_generation_prompt=True``.

    Derivation from ``chat_template.jinja`` in the Qwen3.5 checkpoint:
      * role=tool renders to
        ``<|im_start|>user\\n<tool_response>\\n{content}\\n</tool_response><|im_end|>\\n``
        (the template only emits a fresh ``<|im_start|>user`` when the previous
        message is not also role=tool, which is always true in our loop).
      * ``add_generation_prompt=True`` appends
        ``<|im_start|>assistant\\n<think>\\n`` (thinking is on).
      * A leading ``<|im_end|>\\n`` closes the previous assistant turn — sglang
        stopped at ``</tool_call>`` so the model did not emit ``<|im_end|>``
        itself; the template WOULD have appended it at the turn boundary.

    Why hardcoded instead of calling ``apply_chat_template`` each turn:
    every existing slime multi-turn rollout (retool, search-r1, tau-bench,
    geo3k) uses token-list append. Re-rendering + re-tokenizing per turn
    introduces BPE boundary drift (sglang emits token ids autoregressively;
    tokenizing the decoded conversation as a whole can pick different BPE
    boundaries at seams) plus chat-template whitespace normalization (Qwen
    rewrites ``<think>...</think>...`` with fixed ``\\n`` counts). Both would
    silently desync ``sample.tokens`` from what sglang actually receives.
    See ``notes/QWEN35_CHAT_FORMAT.md`` for details.

    If Qwen3.5 ever changes the tool-response wrapper or the generation
    prompt, update this function. Any smoke test's turn-tail print will
    surface the drift immediately.
    """
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"<tool_response>\n{obs}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


# ---------------------------------------------------------------------------
# Context compression (SUPO session-folding)
# ---------------------------------------------------------------------------

# Verbatim from FoldAgent/agents/prompts.py::SUMMARY_PROMPT_SEARCH. This is
# what the policy is asked to emit when the current sub-trajectory's context
# is about to overflow. The model writes a summary inside <summary>...
# </summary> tags; the summary becomes the ONLY context of the next sub-
# trajectory (plus the original question).
_COMPRESS_PROMPT = """STOP. Your operational context is full and you MUST NOT call any tools in this turn — any <tool_call> will be rejected. Your ONLY job is to generate a concise handover summary by populating the template below. This summary will be your **sole context** for continuing this task. Be brief but ensure all critical data is present.

---

### **`// RESEARCH STATE HANDOVER //`**

**1. Mission Objective**
* **Original Query:** [State the user's verbatim query.]
* **Verification Checklist:**
    * `[Status]` [Checklist Item 1]
    * `[Status]` [Checklist Item 2]
    * ... (List all items with status: `[VERIFIED]`, `[PENDING]`, etc.)

**2. Key Findings**
* [List the most critical, verified facts with sources.]
    * **Fact:** ... **Sources:** [docid)
    * **Fact:** ... **Sources:** [docid)
* **Discrepancies:** [Note any conflicting information found between sources.]

**3. Tactical Plan**
* **Promising Leads:** [List the best remaining keywords, sources, or angles to investigate.]
* **Known Dead Ends:** [List queries or sources that proved useless to avoid repetition.]
* **Immediate Next Action:** [State the exact tool call or query you were about to execute next.]

Now generate the summary, and put your summary inside tag
<summary>
</summary>

REMINDER: Do NOT call any tools. Do NOT continue researching. Write ONLY the <summary>...</summary> block, nothing else."""


def _wrap_summary_request_and_reopen_assistant(prompt: str) -> str:
    """Same shape as _wrap_observation_and_reopen_assistant but for a user turn
    carrying the summary request (not a tool response).

    Byte layout when appended after an existing assistant turn that stopped at
    </tool_call>:
        <|im_end|>\\n        # close the previous assistant turn
        <|im_start|>user\\n{prompt}<|im_end|>\\n
        <|im_start|>assistant\\n<think>\\n
    """
    return (
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


def _extract_summary(text: str) -> str | None:
    """Return the (last) <summary>...</summary> body, stripped, or None.

    Matches SUPO's extract_summary at FoldAgent/agents/fold_agent.py:38.
    """
    if not text:
        return None
    matches = re.findall(r"<summary>(.*?)</summary>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


# Safety margin under rollout_max_context_len so we never send
# `input_ids + max_new_tokens > L2` to sglang and get 400 Bad Request. The
# margin accounts for BPE re-tokenization drift + template overhead + the
# sglang server's own token accounting (which occasionally reports the
# request as a couple tokens longer than we measured). 64 is generous but
# small enough to leave nearly all of the context budget for the model.
_SGLANG_CONTEXT_SAFETY_MARGIN = 64


def _clamp_max_new_tokens(
    args, input_len: int, base_max_new_tokens: int | None = None
) -> int:
    """Return a max_new_tokens value that guarantees sglang won't 400.

    SGLang rejects any request where `len(input_ids) + max_new_tokens`
    exceeds the server's `--context-length` (our L2 = 131072, matches
    `args.rollout_max_context_len` by convention — see
    notes/CONTEXT_LENGTH_LAYERS.md). Without a clamp, once the accumulated
    response grows to `L2 - args.rollout_max_response_len`, EVERY subsequent
    generate call fails with 400 (and eventually blows past the retry cap,
    tearing down the run — observed in RUN qwen3p5-4b-cp2-tp4-20260713-2305-
    multi-iter-v1). Matches retool/coding_agent_rl's per-turn clamp pattern.

    `base_max_new_tokens` defaults to `args.rollout_max_response_len` (the
    normal per-call cap). Returned value is at least 1 so sglang always
    generates something — if headroom is < 1 the caller is about to trigger
    compression anyway, so a short truncated turn is fine.
    """
    if base_max_new_tokens is None:
        base_max_new_tokens = args.rollout_max_response_len
    headroom = args.rollout_max_context_len - input_len - _SGLANG_CONTEXT_SAFETY_MARGIN
    return max(1, min(base_max_new_tokens, headroom))


def _extract_original_question(user_content: str) -> str:
    """Pull the "Question: ..." block out of the parquet user turn so we can
    inline it into the continuation prompt for a compressed sub-trajectory.

    The parquet user template is:
        "You are a deep research agent... Please perform reasoning ...\\n\\n
         Question: {THIS_SAMPLE_QUESTION}\\n\\n
         Your response should contain:\\n..."

    We slice everything between "Question:" and the next double newline that
    starts the "Your response should contain:" preamble. If the pattern
    doesn't match (defensive), fall back to the whole user content.
    """
    m = re.search(r"Question:\s*(.*?)\n\nYour response should contain:", user_content, re.DOTALL)
    if not m:
        return user_content
    return m.group(1).strip()


def _build_continuation_chat(original_prompt: list[dict], summary: str) -> list[dict]:
    """Build the chat message list for a fresh sub-trajectory whose only
    context is the original question + the summary of prior work.

    Reuses the original system prompt (parquet's trimmed task-role blurb)
    verbatim so tools=TOOLS injection is identical to the initial rollout.
    The user turn is a rewritten prompt telling the model it's continuing
    from a summary.
    """
    if not isinstance(original_prompt, list) or len(original_prompt) < 2:
        raise ValueError(f"Expected original_prompt to be a [system, user] list; got {original_prompt!r}")
    system_msg = original_prompt[0]
    user_msg = original_prompt[1]
    question = _extract_original_question(user_msg["content"])
    continuation_user = (
        "You are a deep research agent working on a multi-session task. "
        "You have been researching this question in prior sessions, and "
        "at the end of each session YOU wrote a handover summary to yourself "
        "so this next session could continue. Below is YOUR OWN handover "
        "summary from the last session.\n\n"
        f"Question: {question}\n\n"
        "### Your handover summary (written by you in the last session)\n\n"
        f"{summary}\n\n"
        "Continue researching from where you left off in the summary's "
        "'Immediate Next Action' section.\n\n"
        "Your response should contain:\n"
        "Explanation: {your explanation for your final answer. For this explanation section only, "
        "you should cite your evidence documents inline by enclosing their docids in square "
        "brackets [] at the end of sentences. For example, [20].}\n"
        "Exact Answer: {your succinct, final answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}\n\n"
        "Use finish tool to submit your answer."
    )
    return [
        {"role": "system", "content": system_msg["content"]},
        {"role": "user", "content": continuation_user},
    ]


def _keep_first_n_words(text: str, n: int) -> str:
    if not text:
        return ""
    count = 0
    for m in re.finditer(r"\S+", text):
        count += 1
        if count == n:
            return text[: m.end()] + "\n[Document is truncated.]"
    return text


# ---------------------------------------------------------------------------
# Judge (ported from SUPO reference impl)
# ---------------------------------------------------------------------------

_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def _parse_judge_response(text: str) -> dict:
    result = {"correct": None, "parse_error": False}
    if not text:
        result["parse_error"] = True
        return result
    m = re.search(r"\*\*correct:?\*\*:?\s*(yes|no)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    if m:
        result["correct"] = m.group(1).lower() == "yes"
    else:
        result["parse_error"] = True
    return result


def _norm(s: str) -> str:
    deacc = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = deacc.lower()
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"[“”\"'`]+", "", s)
    s = re.sub(r"[:–—\-_/.,;!()?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _em_score(label: str, pred: str) -> bool:
    ign = {"a", "an", "the", "of", "on", "in", "and", "&", "for", "to", "by", "with"}
    strip = lambda s: re.sub(r"\s+", "", _norm(s))
    toks = lambda s: [t for t in _norm(s).split() if t not in ign and not re.fullmatch(r"\d{4}", t)]
    if strip(label) == strip(pred):
        return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt:
        return False
    if Counter(lt) == Counter(pt):
        return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))):
            return True
    head = lambda s: strip(re.split(r"[:–—-]", _norm(s), 1)[0])
    return head(label) == head(pred)



def _extract_q_dict(s: str) -> dict:
    return {k: v.strip() for k, v in re.findall(r"<(q\d+)>(.*?)</\1>", s, flags=re.S)}


async def _call_openai(messages, model, max_retries=3):
    """Chat completions against Meta's internal MetaGen gateway.

    Goes through the Llama API OpenAI-compatible passthrough (base_url =
    BCPLUS_CONFIGS["judge_base_url"]). Requires LLAMA_API_KEY ("LLM|..." key)
    entitled to the target MetaGen model id.
    """
    from openai import AsyncOpenAI

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    client = AsyncOpenAI(
        api_key=os.environ["LLAMA_API_KEY"],
        base_url=BCPLUS_CONFIGS["judge_base_url"],
    )
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[JUDGE ERROR] {e}"
            await asyncio.sleep(1 * (attempt + 1))
    return ""


def _patch_browsecomp_typos(correct_answer: str, predicted_answer: str) -> tuple[str, str]:
    """Fix three known typos in BrowseComp gold answers before judging.

    BrowseComp (OpenAI's benchmark that BrowseComp-Plus is built on) shipped
    with 3 misspellings in its gold answers. If the model produces the
    correct spelling, exact-match scoring fails and the LLM judge is likely
    to mark it wrong for a "meaningful difference". Rather than mutate the
    parquet (which would break cross-paper comparability), we normalize
    inputs to the judge, matching SUPO's `FoldAgent/envs/local_search.py:214-216`.

    Reversed-string checks avoid this function pattern-matching against
    itself in future edits.
    """
    if "tellomS saiboT"[::-1] in correct_answer:
        correct_answer = correct_answer.replace(
            "tellomS saiboT"[::-1], "ttellomS saiboT"[::-1]
        )
    if "yayhdapattahC najnarawsiB"[::-1] in correct_answer:
        correct_answer = correct_answer.replace(
            "yayhdapattahC najnarawsiB"[::-1], "yayhdapottahC najnarawsiB"[::-1]
        )
    if "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1] in predicted_answer:
        predicted_answer = predicted_answer.replace(
            "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1],
            "yrtnuoC a fo htaP ehT :sedirelC sokfalG"[::-1],
        )
    return correct_answer, predicted_answer


async def _judge_one(question: str, correct_answer: str, predicted_answer: str) -> int:
    correct_answer, predicted_answer = _patch_browsecomp_typos(correct_answer, predicted_answer)
    if _em_score(correct_answer, predicted_answer):
        return 1
    if not predicted_answer.strip():
        return 0
    judge_prompt = _GRADER_TEMPLATE.format(
        question=question, response=predicted_answer, correct_answer=correct_answer
    )
    messages = [{"role": "user", "content": judge_prompt}]
    async with _JUDGE_SEM:
        score = 0
        for _ in range(BCPLUS_CONFIGS["judge_max_retries"]):
            resp = await _call_openai(messages, model=BCPLUS_CONFIGS["judge_model"])
            g = _parse_judge_response(resp)
            if g["parse_error"]:
                continue
            score = int(bool(g["correct"]))
            break
    return score


async def _judge(question: str, correct_answer: str, predicted_answer: str) -> float:
    if "<q1>" in correct_answer:
        gold = _extract_q_dict(correct_answer)
        pred = _extract_q_dict(predicted_answer)
        scores = []
        for k, gv in gold.items():
            if k in pred:
                scores.append(await _judge_one(question, gv, pred[k]))
            else:
                scores.append(0)
        return sum(scores) / len(scores) if scores else 0.0
    return float(await _judge_one(question, correct_answer, predicted_answer))


# ---------------------------------------------------------------------------
# Tool execution: turn a parsed function call into an observation string
# ---------------------------------------------------------------------------


async def _run_action(fn_calls: list[dict], visited: set[str]) -> tuple[str, str | None, bool]:
    """Return (observation_text, finish_answer_or_None, had_effective_call).

    finish_answer_or_None is a non-empty string only if the model called
    <function=finish>. observation_text is what we should append back as the
    user turn (empty string if the trajectory is now terminal).
    had_effective_call is True iff at least one fn in fn_calls dispatched a
    real action (successful search / open_page / non-empty finish). False
    when every call was rejected for bad args, unknown name, or empty finish.
    """
    observation = ""
    finish_answer: str | None = None
    had_effective_call = False
    client = _search_client()
    for fn in fn_calls:
        name = fn["function"]
        args = fn["arguments"]
        if name == "search":
            query = args.get("query", "").strip()
            topk_raw = args.get("topk", BCPLUS_CONFIGS["search_topk_default"])
            try:
                topk = int(topk_raw)
            except (TypeError, ValueError):
                topk = BCPLUS_CONFIGS["search_topk_default"]
            topk = max(1, min(topk, BCPLUS_CONFIGS["search_topk_cap"]))
            if not query:
                observation += '[Error] The "search" function requires a "query" argument.'
                continue
            observation += f'[Search Results for "{query}"]\n'
            async with _SEARCH_SEM:
                # request k=50 upstream then dedup-by-visited like SUPO does
                serp = await client.search(query, 50)
            had_effective_call = True
            shown = 0
            for i, page in enumerate(serp, 1):
                if page["docid"] in visited:
                    text = (
                        "(This page was already seen in a previous search. Here, a shorter snippet is shown. "
                        "If you find this page relevant, please use the open_page tool to inspect the full content) "
                        + " ".join(page["text"].split()[:128])
                    )
                    shown_incr = 0.25
                else:
                    visited.add(page["docid"])
                    text = " ".join(page["text"].split()[: BCPLUS_CONFIGS["doc_words_snippet"]])
                    shown_incr = 1
                observation += (
                    f"\n--- #{i}: {page['docid']}---\n"
                    f"docid: {page['docid']}\n"
                    f"url: {page['url']}\n"
                    f"content: {text}\n"
                )
                shown += shown_incr
                if shown >= topk:
                    break
            observation += "\n"

        elif name == "open_page":
            url = args.get("url")
            docid = args.get("docid")
            if not url and not docid:
                observation += '[Error] The "open_page" function requires either a "docid" or a "url".'
                continue
            async with _SEARCH_SEM:
                opened = await client.open(url=url, docid=docid)
            had_effective_call = True
            for page in opened:
                text = _keep_first_n_words(page.get("text", ""), BCPLUS_CONFIGS["doc_words_full"])
                observation += (
                    "[Opened Page Content]\n"
                    f"docid: {page.get('docid')}\n"
                    f"url: {page.get('url')}\n"
                    f"content: {text}\n"
                )
            observation += "\n"

        elif name == "finish":
            answer = args.get("answer", "")
            if answer.strip():
                finish_answer = answer
                had_effective_call = True
            else:
                observation += (
                    "Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
                    "<tool_call>\n"
                    "<function=finish>\n"
                    "<parameter=answer>\nYOUR ANSWER\n</parameter>\n"
                    "<parameter=explanation>\nYOUR EXPLANATION\n</parameter>\n"
                    "<parameter=confidence>\nYOUR CONFIDENCE\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>\n"
                )

        else:
            observation += f'[Error] The function "{name}" is not supported.'

    if finish_answer is None and observation:
        observation += (
            "\n\n* Please reflect on the information we have obtained, and keep searching for "
            "additional information if we still can not answer the question. Do not give the "
            "answer if the information is still not enough."
        )
    return observation.strip(), finish_answer, had_effective_call


# ---------------------------------------------------------------------------
# The rollout function slime calls
# ---------------------------------------------------------------------------


def _stash_bcplus(sample: Sample, stats: dict) -> None:
    """Populate ``sample.metadata['_bcplus']`` and ``round_number``.

    Called from every exit path of ``_run_one_sub_trajectory`` (normal, length,
    abort, compressed) so the sub-trajectory always carries its stats even on
    early-return paths. ``stats`` fields: n_search, n_open, finished,
    finish_answer, final_stop_reason, n_turns_used, outcome, summary.
    """
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["_bcplus"] = dict(stats)
    # slime's log_multi_turn_data picks this up as multi_turn_metric/round_number_*
    sample.metadata["round_number"] = stats.get("n_turns_used", 0)


async def _do_compression(
    args,
    tokenizer,
    url: str,
    sample: Sample,
    sampling_params: dict,
    prompt_ids: list[int],
    response_token_ids: list[int],
) -> tuple[str | None, str, int]:
    """Run the SUPO summary-generation turn on the current sub-trajectory.

    Appends the summary request + policy summary output to the current
    sub-trajectory's tokens/loss_mask (loss_mask=0 for the request, =1 for
    the model's summary — the model IS learning to summarize).

    Returns ``(summary_or_None, summary_source, extra_tokens_added)``.
    ``summary`` is None only when the model produced literally no output.
    ``summary_source`` is one of:
      * ``"extracted"`` — model emitted a real <summary>...</summary> block
        (the good path).
      * ``"fallback"`` — no <summary> block, but non-empty output; we strip
        the <think> block and salvage the raw text as summary. Downstream
        `reward_post_process` will penalize this sub-traj.
      * ``"empty"`` — model produced literally no output or an abort; no
        salvage possible. Sub-traj will be marked ``compress_failed``.
    ``extra_tokens_added`` is the count added to ``response_token_ids`` this
    call so the caller can update its running length tracker.
    """
    added_tokens_start = len(response_token_ids)

    # Step 1: append the summary request as a user turn (loss_mask=0). Uses
    # the same wrapper shape as tool-response injection except with a plain
    # user message instead of a <tool_response> block.
    request_wrapped = _wrap_summary_request_and_reopen_assistant(_COMPRESS_PROMPT)
    request_tokens = tokenizer(request_wrapped, add_special_tokens=False)["input_ids"]
    response_token_ids.extend(request_tokens)
    sample.append_response_tokens(
        args, tokens=request_tokens, trainable=False, text=request_wrapped,
    )

    # Step 2: sglang generation for the summary. Stop tags:
    #   - </summary>: canonical case, model closed the summary block
    #   - </tool_call>: model chose to emit a tool call instead of a summary
    #     (heuristic-forced compression is against the model's will; we want
    #     to catch it and fall back to using the raw text as summary rather
    #     than letting it burn all remaining budget writing tool calls).
    summary_sampling_params = {
        **sampling_params,
        "stop": ["</summary>", "</tool_call>"],
        # Clamp under L2 headroom (see _clamp_max_new_tokens). Same
        # safeguard as the main-loop turn call — compression is often
        # triggered right at the L2 boundary, so this is the *most*
        # likely call to overflow without clamping.
        "max_new_tokens": _clamp_max_new_tokens(
            args, len(list(prompt_ids) + response_token_ids)
        ),
    }
    current_ids = list(prompt_ids) + response_token_ids
    payload = {
        "input_ids": current_ids,
        "sampling_params": summary_sampling_params,
        "return_logprob": True,
    }
    output = await post(url, payload)

    if output["meta_info"]["finish_reason"]["type"] == "abort":
        # Treat abort during summarization as compression failure.
        return None, "empty", len(response_token_ids) - added_tokens_start

    # TEMP DEBUG (remove after smoke-test verify)
    _summary_text_debug = output["text"]
    print(
        f"[BC+ COMPRESS DEBUG] _do_compression sglang finish_reason={output['meta_info']['finish_reason']['type']!r} "
        f"raw cur_text[:2000]={_summary_text_debug[:2000]!r}",
        flush=True,
    )

    cur_text = output["text"]
    cur_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
    cur_logps = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    response_token_ids.extend(cur_tokens)
    sample.append_response_tokens(
        args,
        tokens=cur_tokens,
        log_probs=cur_logps,
        trainable=True,
        meta_info=output["meta_info"],
        text=cur_text,
    )

    # Extract the summary. Matches SUPO fold_agent.py:139's
    # `summary = extract_summary(response) or response` — if the model didn't
    # emit <summary> tags, fall back to using the entire cur_text as the
    # summary. This handles the common case where the policy under thinking
    # spends its response inside <think>...</think> and never gets to open
    # the <summary> block; we still pass forward *something* the next sub-
    # trajectory can build on. The `summary_source` return tag lets the
    # reward hook penalize this fallback path so the model has a gradient
    # signal to actually learn to emit real <summary> blocks.
    summary = _extract_summary(cur_text)
    if summary is not None:
        summary_source = "extracted"
    elif cur_text.strip():
        # Strip the <think> block if present so the summary is just the visible
        # output the model wrote after thinking.
        without_think = re.sub(r"^.*?</think>\s*", "", cur_text, count=1, flags=re.DOTALL)
        summary = (without_think if without_think.strip() else cur_text).strip()
        summary_source = "fallback"
    else:
        summary_source = "empty"
    return summary, summary_source, len(response_token_ids) - added_tokens_start


async def _run_one_sub_trajectory(
    args,
    sample: Sample,
    sampling_params: dict,
) -> str:
    """Run the ReAct loop for a single sub-trajectory in-place on ``sample``.

    Populates ``sample.tokens``, ``sample.loss_mask``, ``sample.response``,
    ``sample.metadata['_bcplus']`` (via ``_stash_bcplus``), and ``sample.status``.

    Returns an outcome tag: one of ``"finished"``, ``"truncated"``, ``"aborted"``,
    ``"compressed"``, ``"compress_failed"``. The parent ``generate`` uses this
    to decide whether to open a fresh sub-trajectory.

    Loop terminates early with outcome ``"compressed"`` if the response length
    exceeds ``compress_length_threshold * rollout_max_context_len``, in which
    case a summary is generated inline and stashed at
    ``sample.metadata['_bcplus']['summary']``.
    """
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    if not isinstance(sample.prompt, list):
        raise TypeError(
            f"BC+ generate expects sample.prompt to be list[dict]; got {type(sample.prompt)}"
        )
    prompt_text = tokenizer.apply_chat_template(
        sample.prompt, tools=TOOLS, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.response = ""

    response_token_ids: list[int] = []

    # Stop at </tool_call> for tool-call turns (no_stop_trim=True keeps the tag).
    stop_tags = ["</tool_call>"]
    existing_stop = sampling_params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    sampling_params = {
        **sampling_params,
        "stop": list(dict.fromkeys([*existing_stop, *stop_tags])),
    }

    # Compression trigger: full-context length (prompt + accumulated response)
    # exceeds `compress_length_threshold * rollout_max_context_len`. We use
    # rollout_max_context_len (not rollout_max_response_len) because:
    #   * rollout_max_response_len is sglang's per-call max_new_tokens, NOT a
    #     cumulative response cap across turns
    #   * rollout_max_context_len is slime's convention for "total context
    #     budget per sample" — retool/geo3k/coding_agent_rl all use it for
    #     accumulated-length checks (see retool/generate_with_retool.py:244-247)
    # We require it to be set explicitly (no fallback to cp*max_tokens_per_gpu
    # like retool) because a silent fallback risks a wrong-sized budget.
    assert args.rollout_max_context_len is not None, (
        "BC+ compression trigger requires --rollout-max-context-len to be set "
        "explicitly. It caps the total prompt+response budget per sample and "
        "governs when compression fires."
    )
    compress_threshold_tokens = int(
        BCPLUS_CONFIGS["compress_length_threshold"] * args.rollout_max_context_len
    )

    visited_docs: set[str] = set()
    n_search = 0
    n_open = 0
    # Count turns where the model failed to dispatch a valid tool call:
    # parse fail (no <tool_call>), unknown function name, missing required
    # args (search/query, open_page/url|docid), or empty-answer finish.
    # A single turn counts at most once even if _run_action reports multiple
    # errors — we care about "this turn produced zero effective tool calls".
    n_bad_tool_calls = 0
    # Count turns where the search server side failed us in any way — HTTP
    # errors, timeouts, malformed 200 responses (missing keys / bad JSON),
    # client bugs, anything except asyncio.CancelledError. This is the
    # "search server health" signal, kept separate from n_bad_tool_calls so
    # the model-side and server-side failure modes stay distinguishable.
    n_search_server_error = 0
    finish_answer: str | None = None
    finish_reason_last: str | None = None
    summary_from_compress: str | None = None
    summary_source: str = ""  # set by _do_compression on the compression turn
    outcome: str = "unknown"
    _turn = -1

    def _stash():
        _stash_bcplus(sample, {
            "n_search": n_search,
            "n_open": n_open,
            "n_bad_tool_calls": n_bad_tool_calls,
            "n_search_server_error": n_search_server_error,
            "finished": finish_answer is not None,
            "finish_answer": finish_answer or "",
            "final_stop_reason": finish_reason_last or "",
            "n_turns_used": _turn + 1,
            "outcome": outcome,
            "summary": summary_from_compress,
            "summary_source": summary_source,
            "response_len_tokens": len(response_token_ids),
        })

    for _turn in range(BCPLUS_CONFIGS["max_turns"]):
        current_ids = list(prompt_ids) + response_token_ids
        # Clamp max_new_tokens under L2 headroom (see _clamp_max_new_tokens).
        # Without this, once input hits L2 - rollout_max_response_len, every
        # sglang call 400s and the run tears down before compression fires.
        turn_sampling_params = {
            **sampling_params,
            "max_new_tokens": _clamp_max_new_tokens(args, len(current_ids)),
        }
        payload = {
            "input_ids": current_ids,
            "sampling_params": turn_sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        finish_reason = output["meta_info"]["finish_reason"]["type"]
        finish_reason_last = finish_reason
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            outcome = "aborted"
            _stash()
            return outcome

        cur_text = output["text"]
        cur_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_logps = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

        response_token_ids += cur_tokens
        sample.append_response_tokens(
            args,
            tokens=cur_tokens,
            log_probs=cur_logps,
            trainable=True,
            meta_info=output["meta_info"],
            text=cur_text,
        )

        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            outcome = "truncated"
            _stash()
            return outcome

        fn_calls = _extract_fn_call(cur_text)
        if not fn_calls:
            observation = "No function call was detected in the model response."
            n_bad_tool_calls += 1
        else:
            for fn in fn_calls:
                if fn["function"] == "search":
                    n_search += 1
                elif fn["function"] == "open_page":
                    n_open += 1
            try:
                observation, finish_answer, had_effective_call = await _run_action(fn_calls, visited_docs)
            except asyncio.CancelledError:
                # Cooperative cancellation (rollout aborted upstream) — re-raise
                # so the parent task tears down correctly. Not a server error.
                raise
            except Exception as e:
                # Catch-all for anything the search server or client throws:
                # httpx.HTTPError (timeout / conn refused / HTTP 5xx),
                # KeyError / TypeError from malformed 200 responses, JSON
                # decode errors, client bugs, etc. Print so the traceback
                # is grep-able in train.log; increment the counter so wandb
                # (bcplus/n_search_server_error_mean) shows the health.
                print(
                    f"[BCPLUS search-server error] turn={_turn} fn_calls={fn_calls!r} "
                    f"err={type(e).__name__}: {e}",
                    flush=True,
                )
                observation = f"[Search server error] {type(e).__name__}: {e}"
                finish_answer = None
                # Server-side failure is not a model-format error; do not
                # taint n_bad_tool_calls with it. Keep the two signals clean.
                had_effective_call = True
                n_search_server_error += 1
            if not had_effective_call:
                n_bad_tool_calls += 1

        if finish_answer is not None:
            sample.status = Sample.Status.COMPLETED
            outcome = "finished"
            _stash()
            return outcome

        # Compression check BEFORE injecting the tool-response observation.
        # If total context is already over budget, don't blow it further by
        # appending the obs — instead, close out with a compression turn.
        # Total = prompt + accumulated response tokens (this is what sglang
        # sees as input_ids on the next turn).
        total_context_len = len(prompt_ids) + len(response_token_ids)
        if total_context_len > compress_threshold_tokens:
            # TEMP DEBUG (remove after smoke-test verify)
            print(
                f"[BC+ COMPRESS DEBUG] compression triggered on turn {_turn} "
                f"(total_context={total_context_len} = prompt {len(prompt_ids)} + "
                f"resp {len(response_token_ids)} > thresh={compress_threshold_tokens})",
                flush=True,
            )
            summary_from_compress, summary_source, _ = await _do_compression(
                args, tokenizer, url, sample, sampling_params,
                prompt_ids, response_token_ids,
            )
            # TEMP DEBUG
            print(
                f"[BC+ COMPRESS DEBUG] _do_compression returned "
                f"summary_source={summary_source!r} "
                f"summary={'None' if summary_from_compress is None else repr(summary_from_compress[:150]) + '...'}",
                flush=True,
            )
            if summary_from_compress is None:
                sample.status = Sample.Status.TRUNCATED
                outcome = "compress_failed"
            else:
                sample.status = Sample.Status.COMPLETED
                outcome = "compressed"
            _stash()
            return outcome

        obs_wrapped = _wrap_observation_and_reopen_assistant(observation)
        obs_tokens = tokenizer(obs_wrapped, add_special_tokens=False)["input_ids"]
        response_token_ids += obs_tokens
        sample.append_response_tokens(
            args, tokens=obs_tokens, trainable=False, text=obs_wrapped,
        )

    # max_turns exhausted without finish/compress/length/abort.
    sample.status = Sample.Status.TRUNCATED
    outcome = "truncated"
    _stash()
    return outcome


async def generate(args, sample: Sample, sampling_params) -> list[Sample]:
    """Rollout entry point. Returns a LIST of sibling sub-trajectory Samples.

    All siblings share ``rollout_id`` (== the original ``sample.index``) so
    slime's per-rollout loss reducer aggregates them into one rollout, and
    ``build_dp_schedule`` keeps them in the same training step.

    Compression is triggered by ``_run_one_sub_trajectory`` when the current
    sub-trajectory's total context length (prompt + response) exceeds
    ``compress_length_threshold * rollout_max_context_len``. Each triggered
    compression closes the current sub-trajectory (with the summary appended
    to its response tokens) and this loop opens a fresh sub-trajectory whose
    prompt is the original question + the summary.

    Reward is judged only on the FINAL sub-trajectory (the only one that
    could have called ``finish``). Broadcasting the final reward to all
    siblings + GRPO grouping with ``rollout_id`` dedup happens in
    ``reward_post_process``.
    """
    assert not args.partial_rollout, "partial_rollout not supported for BC+ generate."

    parent_rollout_id = sample.index
    original_prompt = sample.prompt
    sub_trajs: list[Sample] = []

    current_sample = sample  # first sub-traj is the input sample itself
    for sub_traj_index in range(BCPLUS_CONFIGS["max_sub_trajs"]):
        current_sample.rollout_id = parent_rollout_id
        # TEMP DEBUG (remove after smoke-test verify)
        print(
            f"[BC+ COMPRESS DEBUG] parent_idx={sample.index} "
            f"sub_traj_index={sub_traj_index} starting sub-traj (rollout_id={parent_rollout_id})",
            flush=True,
        )
        outcome = await _run_one_sub_trajectory(args, current_sample, sampling_params)
        # TEMP DEBUG
        print(
            f"[BC+ COMPRESS DEBUG] parent_idx={sample.index} "
            f"sub_traj_index={sub_traj_index} outcome={outcome!r} "
            f"summary_len={len(current_sample.metadata.get('_bcplus', {}).get('summary') or '')}",
            flush=True,
        )
        sub_trajs.append(current_sample)

        if outcome != "compressed":
            break

        # Compression fired; open the next sub-trajectory with the summary.
        summary = current_sample.metadata["_bcplus"]["summary"]
        if summary is None:  # defensive; outcome == "compressed" implies summary exists
            break

        new_prompt = _build_continuation_chat(original_prompt, summary)
        current_sample = Sample(
            index=sample.index,
            group_index=sample.group_index,
            prompt=new_prompt,
            label=sample.label,
            metadata=copy.deepcopy(sample.metadata) if isinstance(sample.metadata, dict) else {},
        )
        # rollout_id will be set at the top of the next iteration
    else:
        # For-else: max_sub_trajs exhausted while the last outcome was still
        # "compressed". Mark the last sub-traj as TRUNCATED (we bailed on the
        # session because we hit the cap, not because we finished).
        if sub_trajs and sub_trajs[-1].metadata.get("_bcplus", {}).get("outcome") == "compressed":
            sub_trajs[-1].status = Sample.Status.TRUNCATED
            sub_trajs[-1].metadata["_bcplus"]["outcome"] = "compressed_capped"

    # Broadcast sibling metadata so log_bcplus and reward_post_process can find
    # each sub-traj's siblings and identify the final one.
    final = sub_trajs[-1]
    final_bcplus = final.metadata.get("_bcplus", {})
    total = len(sub_trajs)
    for i, s in enumerate(sub_trajs):
        if not isinstance(s.metadata, dict):
            s.metadata = {}
        s.metadata["_bcplus_sibling"] = {
            "sub_traj_index": i,
            "total_sub_trajs": total,
            "is_final": (i == total - 1),
            "final_finish_answer": final_bcplus.get("finish_answer", ""),
            "final_finished": final_bcplus.get("finished", False),
            "parent_rollout_id": parent_rollout_id,
        }

    return sub_trajs


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    """Return a dict of rewards + rollout diagnostics.

    The primary training signal is `score` (0/1 from the judge). Everything else
    is logged to wandb via slime's per-sample dict-reward path so we can watch
    rollout behavior (turn count, tool usage, finish/truncation rate) evolve
    alongside the reward curve.

    Launcher must pass `--reward-key score` so GRPO uses `score` for the
    advantage; the other keys are metrics only.

    Batched calling convention: when the custom generate function returns a
    ``list[Sample]`` (as ours does for compressed rollouts), slime's
    ``batched_async_rm`` (rm_hub/__init__.py:107) hands us the WHOLE list, not
    one sample at a time. We loop and return a parallel list. Single-sample
    calls (from single-Sample generators) still work — we just iterate a
    one-element list.
    """
    if isinstance(sample, list):
        return [await _reward_one(args, s) for s in sample]
    return await _reward_one(args, sample)


async def _reward_one(args, sample: Sample) -> dict:
    """Judge one sample. Splitting the per-sample logic out so ``reward_func``
    can dispatch on Sample vs list[Sample] input."""
    if not isinstance(sample, Sample):
        raise TypeError(f"Sample must be an instance of Sample class; got {type(sample)}")

    md = sample.metadata or {}
    bc = md.get("_bcplus", {}) if isinstance(md, dict) else {}
    finished = bool(bc.get("finished", False))
    finish_answer = bc.get("finish_answer", "") or ""
    n_search = int(bc.get("n_search", 0))
    n_open = int(bc.get("n_open", 0))
    n_bad_tool_calls = int(bc.get("n_bad_tool_calls", 0))
    n_search_server_error = int(bc.get("n_search_server_error", 0))
    n_turns_used = int(bc.get("n_turns_used", 0))
    truncated = int(bc.get("final_stop_reason", "") == "length")

    # `sample.label` is the parquet `answer` column; fall back to metadata.
    gold = sample.label if sample.label is not None else md.get("answer", "")
    question = md.get("query", "") or md.get("problem_statement", "")

    if finished and finish_answer and gold:
        try:
            score = float(await _judge(question, gold, finish_answer))
            judge_failed = 0
        except Exception as e:
            # Never let judge errors kill the training loop.
            print(f"[BCPLUS reward_func] judge failed: {e}")
            score = 0.0
            judge_failed = 1
    else:
        score = 0.0
        judge_failed = 0

    return {
        "score": score,             # primary training signal (used for advantage)
        "n_turns": float(n_turns_used),
        "n_search": float(n_search),
        "n_open": float(n_open),
        "n_tool_calls": float(n_search + n_open),
        "n_bad_tool_calls": float(n_bad_tool_calls),
        "n_search_server_error": float(n_search_server_error),
        "finished": float(int(finished)),
        "truncated": float(truncated),
        "no_finish_score": float(score if not finished else 0),  # marks when 0 came from no-finish
        "judge_failed": float(judge_failed),
    }


def log_bcplus(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Custom rollout log hook wired via ``--custom-rollout-log-function-path``.

    ``rollout_id`` is slime's driver-side step counter (the true wandb
    ``rollout/step`` value). It's passed uniformly for sync and async paths, so
    this hook works identically under both. See notes/rollout_id_data_model.md for
    why this is the right hook rather than reward_post_process — the latter
    receives no rollout_id and its ``sample.rollout_id`` is a
    trajectory-level identifier, not the driver's step counter.

    Handles compressed rollouts: ``samples`` may contain multiple sub-trajs
    per rollout (all sharing ``sample.rollout_id``). Sub-traj-level metrics
    aggregate across all sub-trajs; rollout-level metrics dedupe by
    ``sample.rollout_id`` and use only the FINAL sibling's ``reward`` (that's
    the one carrying the judge score post-broadcast).

    Returns False so slime's default rollout logging still runs afterwards.
    """
    import wandb
    from collections import defaultdict

    if wandb.run is None:
        return False

    flat_samples: list[Sample] = []
    for s in samples:
        if isinstance(s, list):
            flat_samples.extend(s)
        else:
            flat_samples.append(s)

    if not flat_samples:
        return False

    # Group by rollout_id (parent trajectory). Each group is one rollout with
    # 1+ sub-trajs. If sub_traj_sibling metadata is missing (defensive), treat
    # the sample as its own singleton rollout.
    by_rollout: dict[int, list[Sample]] = defaultdict(list)
    for s in flat_samples:
        by_rollout[s.rollout_id if s.rollout_id is not None else id(s)].append(s)

    # Sub-traj-level: aggregate reward diagnostics across ALL sub-trajs.
    diag_keys = ("n_turns", "n_search", "n_open", "n_tool_calls", "n_bad_tool_calls", "n_search_server_error", "finished", "truncated", "judge_failed")
    diag_sums = {k: 0.0 for k in diag_keys}
    diag_count = 0
    for s in flat_samples:
        r = s.reward
        if isinstance(r, dict):
            for k in diag_keys:
                diag_sums[k] += float(r.get(k, 0.0))
            diag_count += 1

    # Rollout-level: dedupe by rollout_id, use final sibling's judged reward
    # for the score metric.
    scores_per_rollout: list[float] = []
    sub_traj_counts: list[int] = []
    compression_fired: list[bool] = []
    compression_failed: list[bool] = []
    capped: list[bool] = []
    final_response_lens: list[int] = []
    # For pass@k: group by group_index (per-prompt), collect per-rollout
    # raw (pre-penalty) score. `raw_final_score` is snapshotted in
    # reward_post_process before the compression-failure penalty is applied,
    # so pass@k reflects pure answer accuracy (matches the traditional
    # "did this rollout get the answer right" semantics), independent of
    # whether the model happened to fail at emitting <summary> blocks.
    group_raw_scores: dict[int, list[float]] = defaultdict(list)
    for _rid, group in by_rollout.items():
        # Find the final sibling (the one with is_final=True, or the last if
        # sibling metadata missing).
        final = next(
            (s for s in group if (s.metadata or {}).get("_bcplus_sibling", {}).get("is_final")),
            group[-1],
        )
        r = final.reward
        if isinstance(r, dict):
            scores_per_rollout.append(float(r.get("score", 0.0)))
        elif r is not None:
            scores_per_rollout.append(float(r))
        sub_traj_counts.append(len(group))
        # Any non-final sibling with outcome="compressed" means at least one
        # compression fired successfully in this rollout.
        outcomes = [(s.metadata or {}).get("_bcplus", {}).get("outcome") for s in group]
        compression_fired.append(any(o in ("compressed", "compressed_capped") for o in outcomes))
        compression_failed.append(any(o == "compress_failed" for o in outcomes))
        capped.append(any(o == "compressed_capped" for o in outcomes))
        final_bcplus = (final.metadata or {}).get("_bcplus", {})
        final_response_lens.append(int(final_bcplus.get("response_len_tokens", 0)))
        # Collect per-group raw scores for pass@k. group_index may be None if
        # something upstream mis-set it — skip those samples (they'd form
        # singleton groups where pass@k is degenerate anyway).
        if final.group_index is not None and "raw_final_score" in final_bcplus:
            group_raw_scores[final.group_index].append(float(final_bcplus["raw_final_score"]))

    n_rollouts = len(by_rollout)

    # Compression-quality metrics (sub-traj level, keyed by summary_source).
    # Denominator for `compress_success_rate` = number of sub-trajs that
    # actually attempted compression (outcome in {"compressed",
    # "compress_failed", "compressed_capped"}). Excludes `finished` /
    # `truncated` / `aborted` sub-trajs that never entered _do_compression.
    # Numerator = sub-trajs where the model emitted a real <summary> block.
    summary_extracted_count = 0
    summary_fallback_count = 0
    attempted_compression_count = 0
    for s in flat_samples:
        bc = (s.metadata or {}).get("_bcplus", {}) if isinstance(s.metadata, dict) else {}
        if bc.get("outcome") in ("compressed", "compress_failed", "compressed_capped"):
            attempted_compression_count += 1
            src = bc.get("summary_source", "")
            if src == "extracted":
                summary_extracted_count += 1
            elif src in ("fallback", "empty"):
                summary_fallback_count += 1

    # Per-sub-traj penalty magnitude (0 when sub-traj wasn't penalized).
    # Averaged over ALL sub-trajs to reflect batch-level penalty pressure.
    compress_penalty = BCPLUS_CONFIGS["compress_penalty"]
    penalty_sum = 0.0
    for s in flat_samples:
        bc = (s.metadata or {}).get("_bcplus", {}) if isinstance(s.metadata, dict) else {}
        if bc.get("outcome") == "compressed" and bc.get("summary_source") != "extracted":
            penalty_sum += compress_penalty

    # pass@k on RAW (pre-penalty) scores. Uses the unbiased combinatorial
    # estimator: for a prompt with n rollouts and c correct, the probability
    # that a random size-k sample contains at least one correct is
    # 1 - C(n-c, k) / C(n, k). Averaged across prompts. "Correct" is
    # `raw_final_score >= 1.0` — strict full-credit (matches BC+ scoring
    # semantics: 1.0 = judge said "yes"; partial multi-question scores like
    # 0.5 count as incorrect for pass@k, which is the traditional definition).
    # k values are powers of 2 up to n_samples_per_prompt (matches slime's
    # compute_pass_rate). Skips prompts with n=1 (pass@k is trivially the
    # answer accuracy there — already covered by score_mean).
    pass_at_k: dict[int, list[float]] = defaultdict(list)
    for group_idx, raw_scores in group_raw_scores.items():
        n = len(raw_scores)
        if n < 2:
            continue
        c = sum(1 for x in raw_scores if x >= 1.0)
        k_values = [2 ** i for i in range(int(math.log2(n)) + 1)]
        for k in k_values:
            # 1 - C(n-c, k) / C(n, k); guard k > n-c → prob 1.
            if k > n - c:
                p = 1.0
            else:
                # math.comb is exact for ints; ratio stays in [0, 1].
                p = 1.0 - math.comb(n - c, k) / math.comb(n, k)
            pass_at_k[k].append(p)

    global _BCPLUS_METRIC_DEFINED
    if not _BCPLUS_METRIC_DEFINED:
        wandb.define_metric("bcplus/*", step_metric="rollout/step")
        _BCPLUS_METRIC_DEFINED = True

    log = {}
    # Sub-traj-averaged diagnostics (each sub-traj counts equally).
    if diag_count > 0:
        for k in diag_keys:
            log[f"bcplus/{k}_mean"] = diag_sums[k] / diag_count
    log["bcplus/n_sub_trajs_total"] = len(flat_samples)

    # Rollout-level.
    log["bcplus/score_mean"] = sum(scores_per_rollout) / n_rollouts if n_rollouts else 0.0
    log["bcplus/score_max"] = max(scores_per_rollout) if scores_per_rollout else 0.0
    log["bcplus/score_hits"] = sum(1 for x in scores_per_rollout if x > 0)
    log["bcplus/n_rollouts"] = n_rollouts
    log["bcplus/n_sub_trajs_mean"] = sum(sub_traj_counts) / n_rollouts if n_rollouts else 0.0
    log["bcplus/n_sub_trajs_max"] = max(sub_traj_counts) if sub_traj_counts else 0
    log["bcplus/compression_rate"] = sum(compression_fired) / n_rollouts if n_rollouts else 0.0
    log["bcplus/compression_failed_rate"] = sum(compression_failed) / n_rollouts if n_rollouts else 0.0
    log["bcplus/compression_capped_rate"] = sum(capped) / n_rollouts if n_rollouts else 0.0
    log["bcplus/final_response_len_mean"] = (
        sum(final_response_lens) / n_rollouts if n_rollouts else 0.0
    )
    # Compression-quality metrics.
    log["bcplus/summary_extracted_count"] = summary_extracted_count
    log["bcplus/summary_fallback_count"] = summary_fallback_count
    log["bcplus/compress_success_rate"] = (
        summary_extracted_count / attempted_compression_count
        if attempted_compression_count else 0.0
    )
    log["bcplus/compress_penalty_mean"] = (
        penalty_sum / len(flat_samples) if flat_samples else 0.0
    )
    # pass@k on raw scores (see computation above). Average across prompts.
    for k in sorted(pass_at_k.keys()):
        log[f"bcplus/pass@{k}_raw"] = sum(pass_at_k[k]) / len(pass_at_k[k])
    log["rollout/step"] = rollout_id
    wandb.log(log)
    return False


def reward_post_process(args, samples):
    """Custom post-process hook wired via ``--custom-reward-post-process-path``.

    Three responsibilities:

    1. **Broadcast final-sibling reward to all siblings**: only the final
       sub-trajectory of each rollout has a ``finish_answer`` for the judge
       to score. Every other sub-traj was closed by compression and has
       ``reward["score"] = 0`` as a placeholder. We overwrite those
       placeholders with the final sibling's judged reward, keyed by
       ``rollout_id``.

    2. **Per-sub-traj compression-failure penalty**: any sub-traj whose
       ``outcome == "compressed"`` and ``summary_source != "extracted"``
       (i.e. compression fired but the model didn't emit a real
       ``<summary>...</summary>`` block, so we fell back to salvaging raw
       text) gets ``BCPLUS_COMPRESS_PENALTY`` subtracted from its score.
       Applied AFTER the group-stats snapshot below, so group mean/std are
       computed from unpenalized per-rollout final rewards — the penalty
       only shifts the failing sub-traj's own advantage, sibling
       advantages are unaffected. ``compressed_capped`` (rollout ran out
       of sub-traj budget) is NOT penalized; only actual compression
       fail-to-emit-summary is.

    3. **GRPO normalization matching SUPO's FOLDGRPO**: group by
       ``group_index`` (the per-prompt id set by slime's data source);
       within each group, dedupe by ``rollout_id`` when collecting rewards
       for mean/std (so a rollout with N sub-trajs contributes exactly one
       reward to the group statistics). Then broadcast the per-rollout
       advantage back to every sub-trajectory of that rollout.

    Denominator invariant: group size == number of distinct rollout_ids in
    the group == ``n_samples_per_prompt``, NEVER sub-trajectory count.

    Backward-compat: when no compression fired, every rollout has one
    sub-traj, so the "dedup by rollout_id" step is a no-op and this reduces
    to standard per-prompt GRPO normalization — matches the previous
    reshape-by-shape logic.
    """
    import torch
    from collections import defaultdict

    flat_samples: list[Sample] = []
    for s in samples:
        if isinstance(s, list):
            flat_samples.extend(s)
        else:
            flat_samples.append(s)

    # Step 1: broadcast final-sibling reward to all siblings of the same
    # rollout_id. We rely on _bcplus_sibling.is_final set by generate().
    rollout_to_final_reward: dict[int, dict | float | None] = {}
    for s in flat_samples:
        sib = (s.metadata or {}).get("_bcplus_sibling", {})
        if sib.get("is_final"):
            rollout_to_final_reward[s.rollout_id] = s.reward
    for s in flat_samples:
        if s.rollout_id in rollout_to_final_reward and rollout_to_final_reward[s.rollout_id] is not None:
            s.reward = copy.deepcopy(rollout_to_final_reward[s.rollout_id])

    # Step 2: raw scalar rewards (in flat_samples order) — POST-broadcast,
    # PRE-penalty. This snapshot is what group stats will be computed from,
    # so penalties don't leak into group mean/std.
    raw_rewards = [s.get_reward_value(args) for s in flat_samples]

    # Snapshot the pre-penalty score into every sibling's metadata so
    # `log_bcplus` can compute pass@k on the un-mutated judge signal
    # (post-penalty scores are the training signal but conflate correctness
    # with compression quality; pass@k should reflect answer accuracy only).
    # Written to every sub-traj — all siblings share the same broadcast
    # reward here, so `raw_final_score` is identical across a rollout.
    for i, s in enumerate(flat_samples):
        if not isinstance(s.metadata, dict):
            s.metadata = {}
        s.metadata.setdefault("_bcplus", {})["raw_final_score"] = float(raw_rewards[i])

    if not (
        args.advantage_estimator in ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline")
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    # Step 3: group by group_index, dedupe by rollout_id. Each rollout
    # contributes ONE reward to its group's mean/std stats even if it
    # produced multiple sub-trajectories. Uses UNPENALIZED raw_rewards so
    # compression-failure penalties don't shift group statistics.
    group_rollout_reward: dict[int, dict[int, float]] = defaultdict(dict)
    for i, s in enumerate(flat_samples):
        if s.group_index is None:
            # Defensive: samples without group_index become their own singleton
            # group (mean=self, std=1) — advantage collapses to 0.
            key = f"_singleton_{i}"
            group_rollout_reward[key][i] = raw_rewards[i]
        else:
            group_rollout_reward[s.group_index][s.rollout_id] = raw_rewards[i]

    # Step 4: apply per-sub-traj compression-failure penalty. Mutates both
    # raw_rewards[i] (used for advantage calc below) and s.reward["score"]
    # (so downstream logging + wandb reflect the penalized value). Only
    # `outcome == "compressed"` counts; `compressed_capped`, `compress_failed`,
    # `finished`, `truncated`, `aborted` are exempt.
    compress_penalty = BCPLUS_CONFIGS["compress_penalty"]
    penalties_applied = 0
    if compress_penalty > 0:
        for i, s in enumerate(flat_samples):
            bc = (s.metadata or {}).get("_bcplus", {}) if isinstance(s.metadata, dict) else {}
            if bc.get("outcome") == "compressed" and bc.get("summary_source") != "extracted":
                pre_score = raw_rewards[i]
                raw_rewards[i] = raw_rewards[i] - compress_penalty
                if isinstance(s.reward, dict) and "score" in s.reward:
                    s.reward["score"] = float(s.reward["score"]) - compress_penalty
                elif not isinstance(s.reward, dict) and s.reward is not None:
                    s.reward = float(s.reward) - compress_penalty
                penalties_applied += 1
                # TEMP DEBUG (remove after smoke-test verify)
                print(
                    f"[BC+ REWARD DEBUG] penalty applied rollout_id={s.rollout_id} "
                    f"summary_source={bc.get('summary_source')!r} "
                    f"pre_score={pre_score:.4f} -> post_score={raw_rewards[i]:.4f}",
                    flush=True,
                )

    # TEMP DEBUG (remove after smoke-test verify)
    print(
        f"[BC+ REWARD DEBUG] n_samples={len(flat_samples)} "
        f"n_groups={len(group_rollout_reward)} "
        f"penalties_applied={penalties_applied} "
        f"group_rollout_reward={dict((k, dict(v)) for k, v in group_rollout_reward.items())}",
        flush=True,
    )

    # Step 5: per-group mean/std; broadcast advantage back to each sub-traj.
    # Uses UNPENALIZED group stats (from step 3 snapshot) but PENALIZED
    # per-sub-traj raw_rewards (from step 4), so the penalty shifts only the
    # failing sub-traj's advantage — sibling advantages unaffected.
    normalized = [0.0] * len(flat_samples)
    use_std = (
        args.advantage_estimator in ("grpo", "gspo", "cispo")
        and args.grpo_std_normalization
    )
    group_stats: dict = {}
    for group_key, rollout_rewards in group_rollout_reward.items():
        vals = torch.tensor(list(rollout_rewards.values()), dtype=torch.float)
        mean = vals.mean().item()
        std = vals.std().item() if len(vals) > 1 else 1.0
        group_stats[group_key] = (mean, std)

    for i, s in enumerate(flat_samples):
        group_key = s.group_index if s.group_index is not None else f"_singleton_{i}"
        mean, std = group_stats[group_key]
        adv = raw_rewards[i] - mean
        if use_std:
            adv = adv / (std + 1e-6)
        normalized[i] = adv

    return raw_rewards, normalized
