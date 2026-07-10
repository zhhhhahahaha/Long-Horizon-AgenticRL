"""Migrate the BrowseComp-Plus parquets from SUPO's bare-<function> tool format
to Qwen3.5's canonical <tool_call><function=...></function></tool_call> format.

What actually changes:
  * `prompt[0]` (system message) content is replaced with the short task-role
    blurb in tool_schemas.QWEN_SYSTEM_PROMPT. The SUPO-style tool teaching block
    is deleted because at rollout time we will call
        tokenizer.apply_chat_template(messages, tools=TOOLS, ...)
    and Qwen3.5's chat template renders the <tools>...</tools> schema block +
    <tool_call> format instructions automatically.
  * `prompt[1]` (user message with the question) is preserved byte-for-byte.
  * All other columns (data_source, ability, reward_model, extra_info, answer)
    are copied through unchanged.

Layout after running (without --dry-run):
    /genai/fsx-project/hhzhang01/datasets/
    ├── BC+_backup/          <-- original SUPO-format parquets, DO NOT USE for training
    │   ├── bc_train.parquet
    │   └── bc_test.parquet
    └── BC+/                 <-- new Qwen3.5-format parquets
        ├── bc_train.parquet
        └── bc_test.parquet

Usage:
    # preview first 3 rows and the fully-rendered chat template output
    python migrate_parquet_to_qwen_tool_format.py --dry-run

    # commit: move originals to BC+_backup/, write new parquets to BC+/
    python migrate_parquet_to_qwen_tool_format.py

Must be run on the login pod (needs /genai access + Qwen3.5-4B tokenizer).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
from transformers import AutoTokenizer

# make `from tool_schemas import ...` work when script is invoked from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tool_schemas import QWEN_SYSTEM_PROMPT, TOOLS

BC_DIR = Path(os.environ.get("BC_DIR", "/genai_hh/datasets/BC+"))
BACKUP_DIR = Path(os.environ.get("BC_BACKUP_DIR", "/genai_hh/datasets/BC+_backup"))
QWEN_CKPT = os.environ.get("QWEN_CKPT", "/genai_hh/models/Qwen3.5-4B")
PARQUET_FILES = ["bc_train.parquet", "bc_test.parquet"]


def rewrite_prompt(prompt: list[dict]) -> list[dict]:
    """Replace the system content, leave everything else alone."""
    if not prompt or prompt[0].get("role") != "system":
        raise ValueError(f"Expected first message role=system, got: {prompt[:1]}")
    new_prompt = list(prompt)
    new_prompt[0] = {**prompt[0], "content": QWEN_SYSTEM_PROMPT}
    return new_prompt


def preview(rows: list[dict], tokenizer, n: int = 3) -> None:
    print("=" * 80)
    print(f"PREVIEW: first {n} rows")
    print("=" * 80)
    for i, row in enumerate(rows[:n]):
        old_prompt = row["prompt"]
        new_prompt = rewrite_prompt(old_prompt)

        old_sys = old_prompt[0]["content"]
        new_sys = new_prompt[0]["content"]
        old_user = old_prompt[1]["content"]
        new_user = new_prompt[1]["content"]

        print(f"\n--- ROW {i} ---")
        print(f"[old system prompt] len={len(old_sys)}, head 200 chars:")
        print(old_sys[:200])
        print("...")
        print()
        print(f"[new system prompt] len={len(new_sys)}, full content:")
        print(new_sys)
        print()

        user_hash_old = hashlib.md5(old_user.encode()).hexdigest()[:12]
        user_hash_new = hashlib.md5(new_user.encode()).hexdigest()[:12]
        assert user_hash_old == user_hash_new, "user turn drift!"
        print(f"[user turn] md5={user_hash_old} (unchanged, len={len(new_user)})")
        print()

        rendered = tokenizer.apply_chat_template(
            new_prompt,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"[apply_chat_template(tools=TOOLS, add_generation_prompt=True)]")
        print(f"total rendered chars: {len(rendered)}")
        print("-" * 80)
        print(rendered)
        print("-" * 80)


def rewrite_table(src: Path) -> pa.Table:
    """Read src parquet, return a new table with rewritten system prompts."""
    table = pq.read_table(src)
    rows = table.to_pylist()
    for row in rows:
        row["prompt"] = rewrite_prompt(row["prompt"])
    return pa.Table.from_pylist(rows, schema=table.schema)


def commit(rows_per_file: dict[str, list[dict]]) -> None:
    if BACKUP_DIR.exists() and any(BACKUP_DIR.iterdir()):
        raise RuntimeError(
            f"{BACKUP_DIR} already exists and is non-empty; refusing to overwrite. "
            "Remove or rename it first if you are re-running the migration."
        )
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for fname in PARQUET_FILES:
        src = BC_DIR / fname
        dst_backup = BACKUP_DIR / fname
        print(f"backup: {src} -> {dst_backup}")
        shutil.move(str(src), str(dst_backup))

    # write BACKUP_DIR/README.md
    (BACKUP_DIR / "README.md").write_text(
        "# BC+ backup — SUPO original format\n\n"
        "These parquets carry the original SUPO paper (arXiv 2510.11967) system "
        "prompt that teaches models to emit bare `<function=...></function>` tool "
        "calls. **Do NOT use them with Qwen3.5**, whose chat template expects "
        "`<tool_call><function=...></function></tool_call>`.\n\n"
        "The active dataset lives at `../BC+/` and was produced by "
        "`examples/supo_browsecomp/scripts/migrate_parquet_to_qwen_tool_format.py`.\n"
    )

    for fname in PARQUET_FILES:
        src_backup = BACKUP_DIR / fname
        dst = BC_DIR / fname
        print(f"rewrite: {src_backup} -> {dst}")
        new_table = rewrite_table(src_backup)
        pq.write_table(new_table, dst)

    # write BC_DIR/README.md
    (BC_DIR / "README.md").write_text(
        "# BC+ (Qwen3.5 tool-call format)\n\n"
        "System prompt has been stripped of the SUPO-style tool teaching block. "
        "Tool schemas (search / open_page / finish) are supplied at rollout time "
        "via `apply_chat_template(messages, tools=[...])`, which triggers "
        "Qwen3.5's built-in `<tools>` block rendering and the "
        "`<tool_call><function=...></function></tool_call>` format instructions.\n\n"
        "See `examples/supo_browsecomp/tool_schemas.py` for the schema list and "
        "`scripts/migrate_parquet_to_qwen_tool_format.py` for the exact transform "
        "that produced these files. The original SUPO parquets are preserved at "
        "`../BC+_backup/`.\n"
    )
    print("done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview first N rows only; do not touch disk.")
    parser.add_argument("--preview-rows", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_CKPT, trust_remote_code=True)

    if args.dry_run:
        for fname in PARQUET_FILES:
            path = BC_DIR / fname
            print(f"\n########## {path} ##########")
            table = pq.read_table(path)
            preview(table.to_pylist(), tokenizer, n=args.preview_rows)
        print("\n[dry-run] no files written.")
        return

    all_rows = {fname: pq.read_table(BC_DIR / fname).to_pylist() for fname in PARQUET_FILES}
    commit(all_rows)


if __name__ == "__main__":
    main()
