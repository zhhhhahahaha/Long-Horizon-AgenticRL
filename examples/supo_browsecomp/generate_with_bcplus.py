"""Multi-turn ReAct rollout + reward for BrowseComp-Plus, following the SUPO
paper's workflow="search" recipe (arXiv 2510.11967).

Two callables are exported and wired into slime via:
    --custom-generate-function-path examples.supo_browsecomp.generate_with_bcplus.generate
    --custom-rm-path                 examples.supo_browsecomp.generate_with_bcplus.reward_func

The BrowseComp-Plus parquet already carries a fully rendered chat prompt (system
prompt with tool descriptions + user turn with the question and few-shot
example), so this file does not re-render prompts. Each rollout turn calls
SGLang /generate, parses the `<function=...><parameter=...></function>` XML,
executes it against the search server (or terminates on `finish`), and feeds the
observation back as a loss-masked user turn until the model finishes / hits
--rollout-max-turns / runs out of budget.

The reward is the OpenAI judge from the SUPO reference implementation
(gpt-4o-mini with a gpt-4.1 fallback for near-miss cases).
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import httpx

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .local_search_client import AsyncSearchClient

BCPLUS_CONFIGS = {
    "max_turns": int(os.environ.get("BCPLUS_MAX_TURNS", "20")),
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
    "judge_fallback_model": os.environ.get("BCPLUS_JUDGE_FALLBACK_MODEL", "gpt-5-4-genai-dss4"),
    "judge_base_url": os.environ.get("BCPLUS_JUDGE_BASE_URL", "https://api.llama.com/compat/v1/"),
    "judge_max_retries": 3,
}

_SEARCH_SEM = asyncio.Semaphore(BCPLUS_CONFIGS["search_concurrency"])
_JUDGE_SEM = asyncio.Semaphore(BCPLUS_CONFIGS["judge_concurrency"])

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
    matches = list(re.finditer(r"(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>", text, re.DOTALL))
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
            "arguments": dict(re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", m.group(2), re.DOTALL)),
        }
        for m in last
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


def _relaxed_em(label: str, pred: str) -> bool:
    strip = lambda s: re.sub(r"\s+", "", _norm(s))
    if not label or not pred:
        return False
    a, b = strip(label), strip(pred)
    if a == b or a in b or b in a:
        return True
    if SequenceMatcher(None, a, b).ratio() >= 0.9:
        return True
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / min(len(a), len(b) or 1) >= 0.9


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


async def _judge_one(question: str, correct_answer: str, predicted_answer: str) -> int:
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
        if score == 0 and _relaxed_em(correct_answer, predicted_answer):
            resp = await _call_openai(messages, model=BCPLUS_CONFIGS["judge_fallback_model"])
            g = _parse_judge_response(resp)
            score = int(bool(g.get("correct")))
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


async def _run_action(fn_calls: list[dict], visited: set[str]) -> tuple[str, str | None]:
    """Return (observation_text, finish_answer_or_None).

    finish_answer_or_None is a non-empty string only if the model called
    <function=finish>. observation_text is what we should append back as the
    user turn (empty string if the trajectory is now terminal).
    """
    observation = ""
    finish_answer: str | None = None
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
            else:
                observation += (
                    "Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
                    "<function=finish>\n"
                    "<parameter=answer>YOUR ANSWER</parameter>\n"
                    "<parameter=explanation>YOUR EXPLANATION</parameter>\n"
                    "<parameter=confidence>YOUR CONFIDENCE</parameter>\n"
                    "</function>\n"
                )

        else:
            observation += f'[Error] The function "{name}" is not supported.'

    if finish_answer is None and observation:
        observation += (
            "\n\n* Please reflect on the information we have obtained, and keep searching for "
            "additional information if we still can not answer the question. Do not give the "
            "answer if the information is still not enough."
        )
    return observation.strip(), finish_answer


# ---------------------------------------------------------------------------
# The rollout function slime calls
# ---------------------------------------------------------------------------


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "partial_rollout not supported for BC+ generate."

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # sample.prompt is a list of chat messages (system + user), rendered to string
    # via the model's chat template. Re-render with an assistant prefix so the
    # model directly generates the assistant turn.
    if isinstance(sample.prompt, list):
        prompt_text = tokenizer.apply_chat_template(
            sample.prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = sample.prompt

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.response = ""

    # Stop the sampler at the tool-call boundary so we can inject an observation
    # without wasting tokens on hallucinated tool responses.
    stop_tags = ["</function>"]
    existing_stop = sampling_params.get("stop") or []
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    sampling_params = {
        **sampling_params,
        "stop": list(dict.fromkeys([*existing_stop, *stop_tags])),
    }

    visited_docs: set[str] = set()
    n_search = 0
    n_open = 0
    finish_answer: str | None = None
    finish_reason_last: str | None = None

    for _turn in range(BCPLUS_CONFIGS["max_turns"]):
        payload = {
            "text": prompt_text + sample.response,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        finish_reason = output["meta_info"]["finish_reason"]["type"]
        finish_reason_last = finish_reason
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_text = output["text"]
        cur_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_logps = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

        sample.append_response_tokens(
            args,
            tokens=cur_tokens,
            log_probs=cur_logps,
            trainable=True,
            meta_info=output["meta_info"],
            text=cur_text,
        )

        if finish_reason == "length":
            break

        fn_calls = _extract_fn_call(cur_text)
        if not fn_calls:
            observation = "No function call was detected in the model response."
        else:
            for fn in fn_calls:
                if fn["function"] == "search":
                    n_search += 1
                elif fn["function"] == "open_page":
                    n_open += 1
            try:
                observation, finish_answer = await _run_action(fn_calls, visited_docs)
            except httpx.HTTPError as e:
                observation = f"[Search server error] {e}"
                finish_answer = None

        if finish_answer is not None:
            break

        # Append the observation to the running response as raw text (loss_mask=0).
        # We wrap it in a lightweight sentinel so the model can distinguish tool
        # output from its own reasoning without having to re-run the chat
        # template every turn. The SUPO reference impl feeds the same wrapping
        # in the {role: user} conversation position; here we inline it because
        # slime's SGLang endpoint is token-appending, not chat-history-diffing.
        obs_wrapped = f"\n\n<observation>\n{observation}\n</observation>\n\n"
        obs_tokens = tokenizer(obs_wrapped, add_special_tokens=False)["input_ids"]
        sample.append_response_tokens(args, tokens=obs_tokens, trainable=False, text=obs_wrapped)

    # Stash rollout stats on the sample so reward_func / wandb can log them.
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["_bcplus"] = {
        "n_search": n_search,
        "n_open": n_open,
        "finished": finish_answer is not None,
        "finish_answer": finish_answer or "",
        "final_stop_reason": finish_reason_last or "",
        "n_turns_used": _turn + 1,
    }
    # slime's log_multi_turn_data picks this up as multi_turn_metric/round_number_*
    sample.metadata["round_number"] = _turn + 1

    match finish_reason_last:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case _:
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> dict:
    """Return a dict of rewards + rollout diagnostics.

    The primary training signal is `score` (0/1 from the judge). Everything else
    is logged to wandb via slime's per-sample dict-reward path so we can watch
    rollout behavior (turn count, tool usage, finish/truncation rate) evolve
    alongside the reward curve.

    Launcher must pass `--reward-key score` so GRPO uses `score` for the
    advantage; the other keys are metrics only.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    md = sample.metadata or {}
    bc = md.get("_bcplus", {}) if isinstance(md, dict) else {}
    finished = bool(bc.get("finished", False))
    finish_answer = bc.get("finish_answer", "") or ""
    n_search = int(bc.get("n_search", 0))
    n_open = int(bc.get("n_open", 0))
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
        "finished": float(int(finished)),
        "truncated": float(truncated),
        "no_finish_score": float(score if not finished else 0),  # marks when 0 came from no-finish
        "judge_failed": float(judge_failed),
    }


def reward_post_process(args, samples):
    """Custom post-process hook: log rollout diagnostics to wandb, then run
    slime's default group-normalized GRPO advantage math.

    Wired via `--custom-reward-post-process-path`. Runs on Ray driver rank so
    it's safe to call wandb.log() directly.
    """
    import torch
    import wandb

    # 1. Aggregate rollout diagnostics from sample.reward dicts (the primary
    #    "score" is one of the fields; others are metrics-only).
    flat_samples = []
    for s in samples:
        if isinstance(s, list):
            flat_samples.extend(s)
        else:
            flat_samples.append(s)

    diag_keys = ("n_turns", "n_search", "n_open", "n_tool_calls", "finished", "truncated", "judge_failed")
    diag_sums = {k: 0.0 for k in diag_keys}
    diag_count = 0
    scores = []
    for s in flat_samples:
        r = s.reward
        if isinstance(r, dict):
            scores.append(float(r.get("score", 0.0)))
            for k in diag_keys:
                diag_sums[k] += float(r.get(k, 0.0))
            diag_count += 1
        else:
            scores.append(float(r) if r is not None else 0.0)

    if diag_count > 0 and wandb.run is not None:
        log = {f"bcplus/{k}_mean": diag_sums[k] / diag_count for k in diag_keys}
        log["bcplus/score_mean"] = sum(scores) / len(scores) if scores else 0.0
        log["bcplus/score_max"] = max(scores) if scores else 0.0
        log["bcplus/score_hits"] = sum(1 for x in scores if x > 0)
        log["bcplus/n_samples"] = diag_count
        wandb.log(log)

    # 2. Standard GRPO group-normalized reward (copy of slime's default logic).
    raw_rewards = [s.get_reward_value(args) for s in flat_samples]
    if (
        args.advantage_estimator in ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline")
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean
        if args.advantage_estimator in ("grpo", "gspo", "cispo") and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)
        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards
