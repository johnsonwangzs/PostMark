"""Deterministic calibration and paired detection metrics."""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from statistics import fmean
from typing import Any

from .common import ConfigurationError, DuplicateIdError, sha256_json


METRICS_VERSION = 1


def _validated_scores(scores: Sequence[float], name: str) -> list[float]:
    values: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ConfigurationError(f"{name} scores must be numeric")
        value = float(score)
        if not math.isfinite(value):
            raise ConfigurationError(f"{name} scores must be finite")
        values.append(value)
    if not values:
        raise ConfigurationError(f"{name} scores cannot be empty")
    return values


def empirical_positive_rate(scores: Sequence[float], threshold: float) -> float:
    values = _validated_scores(scores, "empirical rate")
    if not math.isfinite(threshold):
        raise ConfigurationError("Threshold must be finite")
    return sum(score >= threshold for score in values) / len(values)


def calibration_threshold(
    negative_scores: Sequence[float], *, target_fpr: float = 0.01
) -> tuple[float, float]:
    values = _validated_scores(negative_scores, "calibration negative")
    if (
        isinstance(target_fpr, bool)
        or not isinstance(target_fpr, (int, float))
        or not 0 <= target_fpr <= 1
    ):
        raise ConfigurationError("target_fpr must be in [0, 1]")
    unique_scores = sorted(set(values))
    candidates = sorted(
        set(unique_scores)
        | {math.nextafter(score, math.inf) for score in unique_scores}
    )
    for threshold in candidates:
        if not math.isfinite(threshold):
            continue
        fpr = sum(score >= threshold for score in values) / len(values)
        if fpr <= target_fpr:
            return threshold, fpr
    raise ConfigurationError("No finite calibration threshold satisfies target_fpr")


def roc_auc(negative_scores: Sequence[float], positive_scores: Sequence[float]) -> float:
    negatives = _validated_scores(negative_scores, "held-out negative")
    positives = _validated_scores(positive_scores, "held-out positive")
    ordered_negatives = sorted(negatives)
    wins = 0.0
    for positive in positives:
        lower = bisect_left(ordered_negatives, positive)
        upper = bisect_right(ordered_negatives, positive)
        wins += lower + 0.5 * (upper - lower)
    return wins / (len(positives) * len(negatives))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ConfigurationError("Cannot compute a percentile of no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_statistics(
    negatives: Sequence[float], positives: Sequence[float], threshold: float
) -> dict[str, float]:
    return {
        "roc_auc": roc_auc(negatives, positives),
        "heldout_fpr": empirical_positive_rate(negatives, threshold),
        "heldout_tpr": empirical_positive_rate(positives, threshold),
        "mean_score_delta": fmean(positives) - fmean(negatives),
    }


def evaluate_paired_scores(
    *,
    sample_ids: Sequence[str],
    negative_scores: Sequence[float],
    positive_scores: Sequence[float],
    calibration_ids: Sequence[str],
    calibration_negative_scores: Sequence[float],
    detector_config_sha256: str,
    target_fpr: float = 0.01,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    formal_min_negatives: int = 1000,
    calibration_content_sha256: str | None = None,
    heldout_content_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (
        len(sample_ids) == len(negative_scores) == len(positive_scores)
    ) or not sample_ids:
        raise ConfigurationError("Held-out IDs and paired score arrays must align")
    if len(calibration_ids) != len(calibration_negative_scores) or not calibration_ids:
        raise ConfigurationError("Calibration IDs and scores must align")
    if len(set(sample_ids)) != len(sample_ids):
        raise DuplicateIdError("Duplicate held-out sample ID")
    if len(set(calibration_ids)) != len(calibration_ids):
        raise DuplicateIdError("Duplicate calibration sample ID")
    overlap = set(sample_ids) & set(calibration_ids)
    if overlap:
        raise ConfigurationError(
            f"Calibration and held-out IDs overlap: {sorted(overlap)[:3]}"
        )
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ConfigurationError("bootstrap_samples must be positive")
    if (
        isinstance(formal_min_negatives, bool)
        or not isinstance(formal_min_negatives, int)
        or formal_min_negatives < 1
    ):
        raise ConfigurationError("formal_min_negatives must be positive")

    held = sorted(
        (
            str(sample_id),
            _validated_scores([negative], "held-out negative")[0],
            _validated_scores([positive], "held-out positive")[0],
        )
        for sample_id, negative, positive in zip(
            sample_ids, negative_scores, positive_scores
        )
    )
    calibration = sorted(
        (
            str(sample_id),
            _validated_scores([score], "calibration negative")[0],
        )
        for sample_id, score in zip(calibration_ids, calibration_negative_scores)
    )
    negatives = [item[1] for item in held]
    positives = [item[2] for item in held]
    calibration_scores = [item[1] for item in calibration]
    threshold, calibration_fpr = calibration_threshold(
        calibration_scores, target_fpr=target_fpr
    )
    point = _paired_statistics(negatives, positives, threshold)

    random_generator = random.Random(bootstrap_seed)
    bootstrap_values = {key: [] for key in point}
    for _ in range(bootstrap_samples):
        indices = [random_generator.randrange(len(held)) for _ in held]
        sampled_negatives = [negatives[index] for index in indices]
        sampled_positives = [positives[index] for index in indices]
        statistics = _paired_statistics(
            sampled_negatives, sampled_positives, threshold
        )
        for key, value in statistics.items():
            bootstrap_values[key].append(value)
    confidence_intervals = {
        key: {
            "low": _percentile(values, 0.025),
            "high": _percentile(values, 0.975),
        }
        for key, values in bootstrap_values.items()
    }

    for name, digest in (
        ("calibration_content_sha256", calibration_content_sha256),
        ("heldout_content_sha256", heldout_content_sha256),
    ):
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    calibration_split_sha256 = calibration_content_sha256 or sha256_json(calibration)
    heldout_split_sha256 = heldout_content_sha256 or sha256_json(held)
    evaluation_config = {
        "version": METRICS_VERSION,
        "detector_config_sha256": detector_config_sha256,
        "calibration_split_sha256": calibration_split_sha256,
        "heldout_split_sha256": heldout_split_sha256,
        "threshold_rule": "smallest_finite_tau_with_empirical_fpr_lte_target",
        "decision_rule": "score_gte_tau",
        "target_fpr": float(target_fpr),
        "bootstrap": {
            "method": "paired_id_percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        "formal_min_negatives": formal_min_negatives,
    }
    evaluation_config_sha256 = sha256_json(evaluation_config)
    formal = len(calibration) >= formal_min_negatives and len(held) >= formal_min_negatives
    metrics = {
        "evaluation_config_sha256": evaluation_config_sha256,
        "calibration_split_sha256": calibration_split_sha256,
        "heldout_split_sha256": heldout_split_sha256,
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "calibration_fpr": calibration_fpr,
        "heldout_fpr": point["heldout_fpr"],
        "heldout_tpr": point["heldout_tpr"],
        "roc_auc": point["roc_auc"],
        "mean_score1": fmean(negatives),
        "mean_score2": fmean(positives),
        "mean_score_delta": point["mean_score_delta"],
        "num_calibration_negatives": len(calibration),
        "num_heldout_negatives": len(held),
        "num_heldout_positives": len(held),
        "confidence_intervals": confidence_intervals,
        "metric_status": "formal" if formal else "diagnostic_insufficient_negatives",
    }
    return metrics, evaluation_config
