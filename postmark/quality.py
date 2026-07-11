"""Quality, failure, and coverage aggregation for PostMark outputs."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .common import (
    ConfigurationError,
    DuplicateIdError,
    JsonlError,
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    install_network_guard_from_environment,
    require_offline_environment,
    sha256_json,
    stable_word_count,
)


QUALITY_CONFIG_VERSION = 1
TERMINAL_STATUSES = {"completed", "failed"}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    return result


def _sample_id(record: Mapping[str, Any]) -> str:
    value = record.get("id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise JsonlError("Quality input records require a non-empty string or integer id")
    return str(value)


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise JsonlError(f"{name} must be a list of strings")
    return value


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _requested_presence(text: str, words: Sequence[str]) -> float:
    if not words:
        return 0.0
    lowered = text.lower()
    return sum(word.lower() in lowered for word in words) / len(words)


def _optional_bool(diagnostics: Mapping[str, Any], key: str, default: bool) -> bool:
    value = diagnostics.get(key, default)
    if not isinstance(value, bool):
        raise JsonlError(f"diagnostics.{key} must be boolean")
    return value


def _validate_close(recorded: Any, computed: float, name: str) -> None:
    if recorded is None:
        return
    value = _finite_number(recorded, name)
    if not math.isclose(value, computed, rel_tol=1e-9, abs_tol=1e-12):
        raise ConfigurationError(
            f"{name} disagrees with recomputed value: {value} != {computed}"
        )


def sample_quality_metrics(
    record: Mapping[str, Any],
    *,
    semantic_similarity: float | None = None,
    task_score1: float | None = None,
    task_score2: float | None = None,
) -> dict[str, Any]:
    sample_id = _sample_id(record)
    status = record.get("status")
    if status not in TERMINAL_STATUSES:
        raise JsonlError(f"Quality input id {sample_id!r} is not terminal")
    text1 = record.get("text1")
    text2 = record.get("text2")
    if not isinstance(text1, str) or not isinstance(text2, str):
        raise JsonlError(f"Quality input id {sample_id!r} requires text1/text2 strings")
    list1 = _string_list(record.get("list1"), "list1")
    list2 = _string_list(record.get("list2"), "list2")
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise JsonlError(f"Quality input id {sample_id!r} requires diagnostics")
    eligible_value = record.get("eligible_for_aggregate", True)
    if not isinstance(eligible_value, bool):
        raise JsonlError("eligible_for_aggregate must be boolean")

    list_overlap = _jaccard(list1, list2)
    requested_presence = _requested_presence(text2, list1)
    _validate_close(diagnostics.get("list_overlap"), list_overlap, "diagnostics.list_overlap")
    _validate_close(
        diagnostics.get("requested_word_presence"),
        requested_presence,
        "diagnostics.requested_word_presence",
    )
    length1 = stable_word_count(text1)
    length2 = stable_word_count(text2)
    length_delta = length2 - length1
    if diagnostics.get("length_delta_words") is not None:
        recorded_delta = diagnostics["length_delta_words"]
        if isinstance(recorded_delta, bool) or not isinstance(recorded_delta, int):
            raise JsonlError("diagnostics.length_delta_words must be an integer")
        if recorded_delta != length_delta:
            raise ConfigurationError("diagnostics.length_delta_words is inconsistent")
    relative_length_delta = length_delta / length1 if length1 else None

    groups_value = diagnostics.get("groups", [])
    if not isinstance(groups_value, list) or any(
        not isinstance(group, Mapping) for group in groups_value
    ):
        raise JsonlError("diagnostics.groups must be a list of objects")
    group_thresholds: list[bool] = []
    group_exhausted: list[bool] = []
    for group in groups_value:
        threshold_met = group.get("threshold_met")
        exhausted = group.get("max_attempt_exhausted")
        if not isinstance(threshold_met, bool) or not isinstance(exhausted, bool):
            raise JsonlError("Group threshold/exhaustion flags must be boolean")
        best_presence = _finite_number(
            group.get("best_presence"), "group.best_presence"
        )
        if not 0 <= best_presence <= 1:
            raise ConfigurationError("group.best_presence must be in [0, 1]")
        group_thresholds.append(threshold_met)
        group_exhausted.append(exhausted)
    if "groups" in diagnostics and diagnostics.get("num_groups") is not None:
        num_groups = diagnostics["num_groups"]
        if (
            isinstance(num_groups, bool)
            or not isinstance(num_groups, int)
            or num_groups != len(groups_value)
        ):
            raise ConfigurationError("diagnostics.num_groups is inconsistent")
    if groups_value:
        insertion_success = bool(list1) and bool(text2.strip()) and all(group_thresholds)
        max_attempt_exhausted = any(group_exhausted)
    else:
        insertion_success = _optional_bool(
            diagnostics,
            "insertion_success",
            bool(list1) and bool(text2.strip()) and not bool(diagnostics.get("insertion_failed", True)),
        )
        max_attempt_exhausted = _optional_bool(
            diagnostics, "max_attempt_exhausted", False
        )
    if "insertion_success" in diagnostics and diagnostics["insertion_success"] is not insertion_success:
        raise ConfigurationError("diagnostics.insertion_success is inconsistent")
    if (
        "max_attempt_exhausted" in diagnostics
        and diagnostics["max_attempt_exhausted"] is not max_attempt_exhausted
    ):
        raise ConfigurationError("diagnostics.max_attempt_exhausted is inconsistent")

    empty_output = _optional_bool(diagnostics, "empty_output", False)
    embedding_input_truncated = _optional_bool(
        diagnostics, "embedding_input_truncated", False
    )
    generation_input_truncated = _optional_bool(
        diagnostics, "generation_input_truncated", False
    )
    generation_output_truncated = _optional_bool(
        diagnostics, "generation_output_truncated", False
    )
    failure_reason = diagnostics.get("failure_reason") if status == "failed" else None
    if status == "failed" and not isinstance(failure_reason, str):
        failure_reason = "unspecified_failure"

    semantic_value = (
        None
        if semantic_similarity is None
        else _finite_number(semantic_similarity, "semantic_similarity")
    )
    if semantic_value is not None and not -1 <= semantic_value <= 1:
        raise ConfigurationError("semantic_similarity must be in [-1, 1]")
    task1 = None if task_score1 is None else _finite_number(task_score1, "task_score1")
    task2 = None if task_score2 is None else _finite_number(task_score2, "task_score2")
    if (task1 is None) != (task2 is None):
        raise ConfigurationError("Task score1 and score2 must be provided together")

    return {
        "id": sample_id,
        "eligible_for_aggregate": eligible_value,
        "status": status,
        "failure_reason": failure_reason,
        "insertion_success": insertion_success,
        "max_attempt_exhausted": max_attempt_exhausted,
        "empty_output": empty_output,
        "embedding_input_truncated": embedding_input_truncated,
        "generation_input_truncated": generation_input_truncated,
        "generation_output_truncated": generation_output_truncated,
        "requested_word_presence": requested_presence,
        "list_overlap": list_overlap,
        "text1_words": length1,
        "text2_words": length2,
        "length_delta_words": length_delta,
        "relative_length_delta": relative_length_delta,
        "zero_length_denominator": length1 == 0,
        "semantic_similarity": semantic_value,
        "task_score1": task1,
        "task_score2": task2,
        "task_score_delta": None if task1 is None else task2 - task1,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    finite = [_finite_number(value, "aggregate metric") for value in values]
    return {
        "count": len(finite),
        "mean": fmean(finite),
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
        "p05": _percentile(finite, 0.05),
        "p95": _percentile(finite, 0.95),
    }


def aggregate_quality_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    semantic_scores: Mapping[str, float] | None = None,
    semantic_evaluator_fingerprint: Mapping[str, Any] | None = None,
    task_score1_field: str | None = None,
    task_score2_field: str | None = None,
    task_evaluator_fingerprint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        raise ConfigurationError("Quality aggregation requires at least one record")
    ids = [_sample_id(record) for record in records]
    if len(set(ids)) != len(ids):
        raise DuplicateIdError("Duplicate quality input ID")
    if (semantic_scores is None) != (semantic_evaluator_fingerprint is None):
        raise ConfigurationError("Semantic scores require an evaluator fingerprint")
    unknown_semantic_ids = set(semantic_scores or {}) - set(ids)
    if unknown_semantic_ids:
        raise ConfigurationError("Semantic scores contain unknown sample IDs")
    task_enabled = task_score1_field is not None or task_score2_field is not None
    if task_enabled and (
        not task_score1_field
        or not task_score2_field
        or task_evaluator_fingerprint is None
    ):
        raise ConfigurationError("Task metrics require two fields and evaluator fingerprint")

    sample_metrics: list[dict[str, Any]] = []
    for sample_id, record in zip(ids, records):
        eligible = record.get("eligible_for_aggregate", True) is True
        semantic = None
        if semantic_scores is not None:
            if eligible and sample_id not in semantic_scores:
                raise ConfigurationError(
                    f"Eligible id {sample_id!r} has no semantic similarity score"
                )
            semantic = semantic_scores.get(sample_id)
        task1 = record.get(task_score1_field) if task_enabled else None
        task2 = record.get(task_score2_field) if task_enabled else None
        if task_enabled and eligible and (task1 is None or task2 is None):
            raise ConfigurationError(f"Eligible id {sample_id!r} has incomplete task scores")
        if task_enabled and not eligible and (task1 is None or task2 is None):
            task1 = None
            task2 = None
        sample_metrics.append(
            sample_quality_metrics(
                record,
                semantic_similarity=semantic,
                task_score1=task1,
                task_score2=task2,
            )
        )

    eligible_pairs = [
        (record, metrics)
        for record, metrics in zip(records, sample_metrics)
        if metrics["eligible_for_aggregate"]
    ]
    eligible_count = len(eligible_pairs)
    ineligible_count = len(records) - eligible_count
    if eligible_count + ineligible_count != len(records):
        raise ConfigurationError("Quality coverage counts do not conserve input records")
    selection_hashes = {
        record.get("selection_config_sha256") for record, _ in eligible_pairs
    }
    run_hashes = {record.get("run_config_sha256") for record, _ in eligible_pairs}
    if eligible_count and (
        None in selection_hashes
        or None in run_hashes
        or len(selection_hashes) != 1
        or len(run_hashes) != 1
    ):
        raise ConfigurationError("Eligible quality records mix or omit configuration hashes")

    eligible_metrics = [metrics for _, metrics in eligible_pairs]
    status_counts = Counter(metrics["status"] for metrics in eligible_metrics)
    if sum(status_counts.values()) != eligible_count:
        raise ConfigurationError("Terminal status counts do not conserve eligible records")
    failure_counts = Counter(
        metrics["failure_reason"]
        for metrics in eligible_metrics
        if metrics["status"] == "failed"
    )
    if sum(failure_counts.values()) != status_counts.get("failed", 0):
        raise ConfigurationError("Failure classes do not conserve failed records")

    boolean_fields = (
        "insertion_success",
        "max_attempt_exhausted",
        "empty_output",
        "embedding_input_truncated",
        "generation_input_truncated",
        "generation_output_truncated",
        "zero_length_denominator",
    )
    rates = {
        field: (
            sum(metrics[field] for metrics in eligible_metrics) / eligible_count
            if eligible_count
            else None
        )
        for field in boolean_fields
    }
    distributions = {
        field: _distribution(
            [
                metrics[field]
                for metrics in eligible_metrics
                if metrics[field] is not None
            ]
        )
        for field in (
            "requested_word_presence",
            "list_overlap",
            "text1_words",
            "text2_words",
            "length_delta_words",
            "relative_length_delta",
            "semantic_similarity",
            "task_score1",
            "task_score2",
            "task_score_delta",
        )
    }
    input_content_sha256 = sha256_json(
        sorted((sample_id, sha256_json(record)) for sample_id, record in zip(ids, records))
    )
    quality_config = {
        "version": QUALITY_CONFIG_VERSION,
        "input_content_sha256": input_content_sha256,
        "selection_config_sha256": next(iter(selection_hashes), None),
        "run_config_sha256": next(iter(run_hashes), None),
        "semantic_evaluator_fingerprint": (
            dict(semantic_evaluator_fingerprint)
            if semantic_evaluator_fingerprint is not None
            else None
        ),
        "task_evaluator_fingerprint": (
            dict(task_evaluator_fingerprint)
            if task_evaluator_fingerprint is not None
            else None
        ),
        "task_score_fields": (
            {"negative": task_score1_field, "positive": task_score2_field}
            if task_enabled
            else None
        ),
        "aggregation": {
            "denominator": "eligible_ids",
            "percentiles": [0.05, 0.5, 0.95],
            "clean_condition_only": True,
            "failed_samples_retained": True,
        },
    }
    report = {
        "quality_config_sha256": sha256_json(quality_config),
        "quality_config": quality_config,
        "coverage": {
            "total_input": len(records),
            "eligible": eligible_count,
            "ineligible": ineligible_count,
            "status_counts": dict(sorted(status_counts.items())),
            "failure_class_counts": dict(sorted(failure_counts.items())),
        },
        "rates": rates,
        "distributions": distributions,
        "semantic_evaluator_fingerprint": quality_config[
            "semantic_evaluator_fingerprint"
        ],
        "task_evaluator_fingerprint": quality_config["task_evaluator_fingerprint"],
        "clean_condition_only": True,
    }
    return report, sample_metrics


def write_quality_report(
    *,
    input_path: str,
    output_path: str,
    sample_output_path: str | None = None,
    semantic_scores: Mapping[str, float] | None = None,
    semantic_evaluator_fingerprint: Mapping[str, Any] | None = None,
    task_score1_field: str | None = None,
    task_score2_field: str | None = None,
    task_evaluator_fingerprint: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    sample_output = Path(sample_output_path) if sample_output_path else None
    if not overwrite and (
        output.exists() or (sample_output is not None and sample_output.exists())
    ):
        raise ConfigurationError("Quality output already exists; use --overwrite")
    records = load_jsonl(input_path)
    report, samples = aggregate_quality_metrics(
        records,
        semantic_scores=semantic_scores,
        semantic_evaluator_fingerprint=semantic_evaluator_fingerprint,
        task_score1_field=task_score1_field,
        task_score2_field=task_score2_field,
        task_evaluator_fingerprint=task_evaluator_fingerprint,
    )
    atomic_write_json(output, report)
    if sample_output is not None:
        atomic_write_jsonl(sample_output, samples)
    return report


class NomicSemanticEvaluator:
    """Local cosine-similarity evaluator explicitly labeled as a Nomic proxy."""

    def __init__(self, encoder: Any, *, encoder_contract: Mapping[str, Any]) -> None:
        if not callable(getattr(encoder, "encode_texts", None)):
            raise ConfigurationError("Semantic evaluator requires encode_texts")
        self.encoder = encoder
        self.config = {
            "version": 1,
            "evaluator_type": "nomic_proxy",
            "metric": "cosine_similarity",
            "empty_text_policy": "score_zero",
            "encoder_contract": dict(encoder_contract),
        }
        self.fingerprint_sha256 = sha256_json(self.config)

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {**self.config, "sha256": self.fingerprint_sha256}

    def score_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        import torch

        scores: dict[str, float] = {}
        nonempty: list[tuple[str, str, str]] = []
        seen_ids: set[str] = set()
        for record in records:
            sample_id = _sample_id(record)
            if sample_id in seen_ids:
                raise DuplicateIdError(f"Duplicate semantic evaluator ID {sample_id!r}")
            seen_ids.add(sample_id)
            if record.get("eligible_for_aggregate", True) is not True:
                continue
            text1 = record.get("text1")
            text2 = record.get("text2")
            if not isinstance(text1, str) or not isinstance(text2, str):
                raise JsonlError(
                    f"Semantic evaluator id {sample_id!r} requires text1/text2 strings"
                )
            if not text1.strip() or not text2.strip():
                scores[sample_id] = 0.0
            else:
                nonempty.append((sample_id, text1, text2))
        if nonempty:
            left = self.encoder.encode_texts([item[1] for item in nonempty])
            right = self.encoder.encode_texts([item[2] for item in nonempty])
            if (
                left.ndim != 2
                or right.ndim != 2
                or left.shape != right.shape
                or left.shape[0] != len(nonempty)
                or not torch.isfinite(left).all()
                or not torch.isfinite(right).all()
            ):
                raise ConfigurationError(
                    "Nomic semantic evaluator returned invalid embeddings"
                )
            left = torch.nn.functional.normalize(left.to(torch.float32), p=2, dim=1)
            right = torch.nn.functional.normalize(right.to(torch.float32), p=2, dim=1)
            similarities = (left * right).sum(dim=1).detach().cpu().tolist()
            for (sample_id, _, _), similarity in zip(nonempty, similarities):
                scores[sample_id] = max(
                    -1.0,
                    min(
                        1.0,
                        _finite_number(similarity, "Nomic semantic similarity"),
                    ),
                )
        return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate PostMark quality, failure, and coverage metrics."
    )
    parser.add_argument("--config", default="configs/postmark_portable.json")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sample_output_path")
    parser.add_argument(
        "--semantic_evaluator",
        choices=("nomic_proxy", "none"),
        default="nomic_proxy",
    )
    parser.add_argument("--embedder_path")
    parser.add_argument("--embedder_tokenizer_path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--task_score1_field")
    parser.add_argument("--task_score2_field")
    parser.add_argument("--task_evaluator_name")
    parser.add_argument("--task_evaluator_sha256")
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_offline_environment()
    if not args.local_files_only:
        raise ConfigurationError("PostMark-Local requires --local_files_only")
    records = load_jsonl(args.input_path)
    semantic_scores = None
    semantic_fingerprint = None
    if args.semantic_evaluator == "nomic_proxy":
        from .config import PostMarkConfig
        from .nomic_embedder import NomicTextEncoder

        install_network_guard_from_environment()

        config_path = Path(args.config).resolve()
        project_root = config_path.parent.parent
        config = PostMarkConfig.load(config_path)
        paths = config.paths.resolved(project_root)
        encoder = NomicTextEncoder(
            args.embedder_path or str(paths["embedder"]),
            tokenizer_path=(
                args.embedder_tokenizer_path
                or str(paths["embedder_tokenizer"])
            ),
            max_length=config.embedding.max_length,
            task_prefix=config.embedding.task_prefix,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=True,
        )
        evaluator = NomicSemanticEvaluator(
            encoder,
            encoder_contract={
                "model_fingerprint": encoder.model_fingerprint().to_dict(),
                "tokenizer_fingerprint": encoder.tokenizer_fingerprint().to_dict(),
                "pooling": config.embedding.pooling,
                "normalization": config.embedding.normalization,
                "max_length": config.embedding.max_length,
                "task_prefix": config.embedding.task_prefix,
            },
        )
        semantic_scores = evaluator.score_records(records)
        semantic_fingerprint = evaluator.fingerprint
    else:
        install_network_guard_from_environment()

    task_enabled = args.task_score1_field or args.task_score2_field
    task_fingerprint = None
    if task_enabled:
        digest = args.task_evaluator_sha256
        if (
            not args.task_score1_field
            or not args.task_score2_field
            or not args.task_evaluator_name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ConfigurationError(
                "Task metrics require both fields, evaluator name, and lowercase SHA-256"
            )
        task_fingerprint = {
            "name": args.task_evaluator_name,
            "sha256": digest,
        }

    output = Path(args.output_path)
    sample_output = Path(args.sample_output_path) if args.sample_output_path else None
    if not args.overwrite and (
        output.exists() or (sample_output is not None and sample_output.exists())
    ):
        raise ConfigurationError("Quality output already exists; use --overwrite")
    report, samples = aggregate_quality_metrics(
        records,
        semantic_scores=semantic_scores,
        semantic_evaluator_fingerprint=semantic_fingerprint,
        task_score1_field=args.task_score1_field,
        task_score2_field=args.task_score2_field,
        task_evaluator_fingerprint=task_fingerprint,
    )
    atomic_write_json(output, report)
    if sample_output is not None:
        atomic_write_jsonl(sample_output, samples)
    print(
        f"total={report['coverage']['total_input']} "
        f"eligible={report['coverage']['eligible']} "
        f"ineligible={report['coverage']['ineligible']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
