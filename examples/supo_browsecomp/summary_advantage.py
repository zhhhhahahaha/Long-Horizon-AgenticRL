"""Token-level advantage overrides for BrowseComp-Plus summary turns."""

from __future__ import annotations

import os

import torch


def compute_summary_aware_advantages(args, rollout_data) -> None:
    """Build GRPO advantages and override malformed summary-turn tokens.

    ``reward_post_process`` has already converted each parent rollout's final
    judge score into a scalar base advantage and broadcast it to all sibling
    sub-trajectories. This function expands that scalar over response tokens.
    For a fallback compression turn, only the model-generated compression span
    (thinking plus visible output) is replaced with a fixed negative advantage.
    """
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    rewards = rollout_data["rewards"]
    kl = rollout_data["kl"]
    metadata = rollout_data.get("metadata") or [{} for _ in rewards]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]

    expected = len(rewards)
    lengths = {
        "kl": len(kl),
        "metadata": len(metadata),
        "total_lengths": len(total_lengths),
        "response_lengths": len(response_lengths),
    }
    mismatched = {key: value for key, value in lengths.items() if value != expected}
    if mismatched:
        raise ValueError(f"BC+ summary advantage batch length mismatch: rewards={expected}, {mismatched}")

    penalty = float(os.environ.get("BCPLUS_COMPRESS_PENALTY", "0.5"))
    advantages: list[torch.Tensor] = []

    for i, (base_reward, kl_i, md, total_length, response_length) in enumerate(
        zip(rewards, kl, metadata, total_lengths, response_lengths, strict=True)
    ):
        advantage = torch.full_like(kl_i, float(base_reward))
        md = md if isinstance(md, dict) else {}

        if penalty > 0 and md.get("summary_source") == "fallback":
            start = md.get("summary_turn_start")
            end = md.get("summary_turn_end")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(
                    f"BC+ fallback summary sample {i} is missing integer summary-turn span: "
                    f"start={start!r}, end={end!r}"
                )
            if not 0 <= start < end <= response_length:
                raise ValueError(
                    f"BC+ fallback summary sample {i} has invalid response-relative span "
                    f"[{start}, {end}) for response_length={response_length}"
                )

            full_summary_mask = torch.zeros(response_length, dtype=torch.bool, device=advantage.device)
            full_summary_mask[start:end] = True
            local_summary_mask = slice_log_prob_with_cp(
                full_summary_mask,
                int(total_length),
                int(response_length),
            )
            if local_summary_mask.shape != advantage.shape:
                raise ValueError(
                    f"BC+ summary mask shape mismatch for sample {i}: "
                    f"mask={tuple(local_summary_mask.shape)}, advantage={tuple(advantage.shape)}"
                )
            advantage = torch.where(
                local_summary_mask,
                torch.full_like(advantage, -penalty),
                advantage,
            )

        advantages.append(advantage)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]
