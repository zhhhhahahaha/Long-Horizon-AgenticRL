"""Deterministic adaptive sampling for BrowseComp-Plus GRPO training."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
    from slime.utils.types import Sample


logger = logging.getLogger(__name__)

_ALLOWED_ZERO_STD_GROUPS = 2
_TOPUP_CONFIDENCE = 0.95
_TOPUP_GRANULARITY = 8
_REWARD_EPS = 1e-6


@dataclass(frozen=True)
class _CandidateGroup:
    group: list[Any]
    from_first_pool: bool
    index: int
    rewards: tuple[float, ...]

    @property
    def has_nonzero_std(self) -> bool:
        return max(self.rewards) - min(self.rewards) > _REWARD_EPS

    @property
    def is_all_correct(self) -> bool:
        return all(abs(reward - 1.0) <= _REWARD_EPS for reward in self.rewards)

    @property
    def is_all_wrong(self) -> bool:
        return all(abs(reward) <= _REWARD_EPS for reward in self.rewards)


def _final_sample(rollout: Sample | list[Sample]) -> Sample:
    if not isinstance(rollout, list):
        return rollout
    if not rollout:
        raise ValueError("BC+ dynamic sampling received an empty fan-out rollout")

    finals = [
        sample
        for sample in rollout
        if isinstance(sample.metadata, dict) and sample.metadata.get("_bcplus_sibling", {}).get("is_final")
    ]
    if len(finals) > 1:
        raise ValueError(f"BC+ fan-out rollout has {len(finals)} final siblings; expected at most one")
    return finals[0] if finals else rollout[-1]


def _candidate_group(args, group: list[Any], *, from_first_pool: bool) -> _CandidateGroup:
    if len(group) != args.n_samples_per_prompt:
        raise ValueError(f"BC+ candidate group has {len(group)} rollouts; expected {args.n_samples_per_prompt}")

    final_samples = [_final_sample(rollout) for rollout in group]
    if final_samples[0].index is None:
        raise ValueError("BC+ candidate group is missing its sample index")

    rewards = tuple(float(sample.get_reward_value(args)) for sample in final_samples)
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError(f"BC+ candidate group has non-finite rewards: {rewards}")

    return _CandidateGroup(
        group=group,
        from_first_pool=from_first_pool,
        index=int(final_samples[0].index),
        rewards=rewards,
    )


def _beta_binomial_tail_probability(
    *,
    num_trials: int,
    min_successes: int,
    alpha: float,
    beta: float,
) -> float:
    """Return P(X >= min_successes) for a Beta-Binomial random variable."""
    if min_successes <= 0:
        return 1.0
    if min_successes > num_trials:
        return 0.0

    log_beta_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_probabilities = []
    for successes in range(min_successes, num_trials + 1):
        failures = num_trials - successes
        log_combination = math.lgamma(num_trials + 1) - math.lgamma(successes + 1) - math.lgamma(failures + 1)
        log_beta_posterior = (
            math.lgamma(successes + alpha) + math.lgamma(failures + beta) - math.lgamma(num_trials + alpha + beta)
        )
        log_probabilities.append(log_combination + log_beta_posterior - log_beta_prior)

    max_log_probability = max(log_probabilities)
    probability = math.exp(max_log_probability) * sum(
        math.exp(value - max_log_probability) for value in log_probabilities
    )
    return min(max(probability, 0.0), 1.0)


def _choose_topup_group_count(
    *,
    first_pool_valid_count: int,
    first_pool_group_count: int,
    target_valid_count: int,
    confidence: float = _TOPUP_CONFIDENCE,
    granularity: int = _TOPUP_GRANULARITY,
    max_topup_group_count: int,
) -> int:
    if not 0 <= first_pool_valid_count <= first_pool_group_count:
        raise ValueError("BC+ dynamic sampling valid count must be within the first-pool size")
    if not 0 < confidence <= 1:
        raise ValueError("BC+ dynamic sampling confidence must be in (0, 1]")

    missing_valid_count = target_valid_count - first_pool_valid_count
    if missing_valid_count <= 0:
        return 0
    if granularity <= 0 or max_topup_group_count <= 0:
        raise ValueError("BC+ dynamic sampling top-up granularity and cap must be positive")

    alpha = first_pool_valid_count + 1
    beta = first_pool_group_count - first_pool_valid_count + 1
    candidate_sizes = list(range(granularity, max_topup_group_count + 1, granularity))
    if not candidate_sizes or candidate_sizes[-1] != max_topup_group_count:
        candidate_sizes.append(max_topup_group_count)

    probability = 0.0
    for candidate_size in candidate_sizes:
        probability = _beta_binomial_tail_probability(
            num_trials=candidate_size,
            min_successes=missing_valid_count,
            alpha=alpha,
            beta=beta,
        )
        if probability >= confidence:
            return candidate_size

    logger.warning(
        "BC+ dynamic sampling cannot reach predictive confidence %.3f within top-up cap %s "
        "(first_valid=%s/%s, target=%s, capped_probability=%.3f)",
        confidence,
        max_topup_group_count,
        first_pool_valid_count,
        first_pool_group_count,
        target_valid_count,
        probability,
    )
    return max_topup_group_count


def _sampling_targets(batch_size: int) -> tuple[int, int, int]:
    if batch_size <= 0:
        raise ValueError("BC+ dynamic sampling requires a positive rollout batch size")
    first_pool_group_count = 2 * batch_size
    target_valid_count = max(batch_size - _ALLOWED_ZERO_STD_GROUPS, 0)
    max_topup_group_count = first_pool_group_count
    return first_pool_group_count, target_valid_count, max_topup_group_count


def _select_candidates(candidates: list[_CandidateGroup], batch_size: int) -> list[_CandidateGroup]:
    ordered = sorted(candidates, key=lambda candidate: candidate.index)
    valid = [candidate for candidate in ordered if candidate.has_nonzero_std]
    zero_std = [candidate for candidate in ordered if not candidate.has_nonzero_std]
    selected = sorted((valid + zero_std)[:batch_size], key=lambda candidate: candidate.index)
    if len(selected) != batch_size:
        raise RuntimeError(f"BC+ dynamic sampling selected {len(selected)} groups; expected {batch_size}")
    return selected


def _build_metrics(
    all_candidates: list[_CandidateGroup],
    selected_candidates: list[_CandidateGroup],
    topup_group_count: int,
) -> dict[str, int]:
    return {
        "dynamic_sampling/candidate_zero_std_1_count": sum(candidate.is_all_correct for candidate in all_candidates),
        "dynamic_sampling/candidate_zero_std_0_count": sum(candidate.is_all_wrong for candidate in all_candidates),
        "dynamic_sampling/topup_requested_group_count": topup_group_count,
        "dynamic_sampling/first_pool_kept_group_count": sum(
            candidate.from_first_pool for candidate in selected_candidates
        ),
        "dynamic_sampling/selected_group_count": len(selected_candidates),
    }


async def _generate_candidate_groups(args, groups: list[list[Sample]]) -> list[list[Any]]:
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group

    state = GenerateState(args)
    tasks = [
        asyncio.create_task(
            generate_and_rm_group(
                args,
                group,
                sampling_params=state.sampling_params.copy(),
                evaluation=False,
            )
        )
        for group in groups
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _generate_rollout_async(args, data_source) -> RolloutFnTrainOutput:
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.misc import load_function

    assert args.rollout_global_dataset
    assert not args.partial_rollout, "Partial rollout is not supported by BC+ dynamic sampling"

    batch_size = args.rollout_batch_size
    first_pool_group_count, target_valid_count, max_topup_group_count = _sampling_targets(batch_size)
    state = GenerateState(args)
    state.reset()

    try:
        first_groups = await _generate_candidate_groups(
            args,
            data_source.get_samples(first_pool_group_count),
        )
        first_candidates = [_candidate_group(args, group, from_first_pool=True) for group in first_groups]
        first_valid_count = sum(candidate.has_nonzero_std for candidate in first_candidates)

        topup_group_count = _choose_topup_group_count(
            first_pool_valid_count=first_valid_count,
            first_pool_group_count=first_pool_group_count,
            target_valid_count=target_valid_count,
            max_topup_group_count=max_topup_group_count,
        )
        topup_candidates: list[_CandidateGroup] = []
        if topup_group_count:
            topup_groups = await _generate_candidate_groups(
                args,
                data_source.get_samples(topup_group_count),
            )
            topup_candidates = [_candidate_group(args, group, from_first_pool=False) for group in topup_groups]

        all_candidates = sorted(first_candidates + topup_candidates, key=lambda candidate: candidate.index)
        selected_candidates = _select_candidates(all_candidates, batch_size)

        selected_groups = [candidate.group for candidate in selected_candidates]
        if args.rollout_sample_filter_path is not None:
            load_function(args.rollout_sample_filter_path)(args, selected_groups)
        if args.rollout_all_samples_process_path is not None:
            load_function(args.rollout_all_samples_process_path)(
                args,
                [candidate.group for candidate in all_candidates],
                data_source.get_samples,
            )

        metrics = _build_metrics(all_candidates, selected_candidates, topup_group_count)
        logger.info(
            "BC+ dynamic sampling: first_pool=%s first_valid=%s topup=%s selected=%s",
            first_pool_group_count,
            first_valid_count,
            topup_group_count,
            len(selected_candidates),
        )
        return RolloutFnTrainOutput(samples=selected_groups, metrics=metrics)
    finally:
        state.reset()


def generate_rollout(
    args,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout

        return default_generate_rollout(args, rollout_id, data_source, evaluation=True)

    from slime.utils.async_utils import run

    return run(_generate_rollout_async(args, data_source))
