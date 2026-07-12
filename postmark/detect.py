"""Blind PostMark-Local detection using a local keyed selector resource."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .common import (
    ConfigurationError,
    DuplicateIdError,
    JsonlError,
    ResourceMismatchError,
    SelectionError,
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    install_network_guard_from_environment,
    require_offline_environment,
    sha256_json,
    stable_content_id,
)
from .config import PostMarkConfig
from .metrics import evaluate_paired_scores
from .selection_policy import SelectionPolicy


DETECTOR_CONFIG_VERSION = 1


class DetectorSelector(Protocol):
    selection_config_sha256: str
    selection_config: Mapping[str, Any]
    table_manifest: Any
    config_consistent: bool
    eligible_for_aggregate: bool

    def word_count_to_k(self, text: str) -> int: ...

    def select_words(self, text: str, *, top_k: int | None = None) -> list[str]: ...


class PresenceScorer(Protocol):
    fingerprint: Mapping[str, Any]
    fingerprint_sha256: str

    def score(self, text: str, expected_words: Sequence[str]) -> Any: ...


def _selector_resource_sha256(selector: DetectorSelector) -> str:
    value = getattr(selector.table_manifest, "content_sha256", None)
    if not isinstance(value, str) or not value:
        raise ConfigurationError("Selector has no resource content fingerprint")
    return value


def _presence_resource_sha256(presence: PresenceScorer) -> str:
    fingerprint = presence.fingerprint
    if fingerprint.get("presence_mode") != "exact_lemma":
        return presence.fingerprint_sha256
    model = fingerprint.get("spacy_model")
    if isinstance(model, Mapping):
        resource = model.get("fingerprint")
        if isinstance(resource, Mapping) and isinstance(resource.get("sha256"), str):
            return resource["sha256"]
    return presence.fingerprint_sha256


class BlindPostMarkDetector:
    def __init__(
        self,
        selector: DetectorSelector,
        presence: PresenceScorer,
        *,
        detector_profile: str = "portable",
        presence_mode: str = "exact_lemma",
        min_watermark_words: int | None = None,
        max_watermark_words: int | None = None,
    ) -> None:
        if detector_profile != "portable":
            raise ConfigurationError("This detector currently supports portable profile only")
        if presence_mode not in {"exact_lemma", "nomic_fuzzy"}:
            raise ConfigurationError("Unsupported portable presence mode")
        if presence.fingerprint.get("presence_mode") != presence_mode:
            raise ConfigurationError("Presence scorer mode does not match detector configuration")
        self.selector = selector
        self.presence = presence
        self.detector_profile = detector_profile
        self.presence_mode = presence_mode
        self.selection_policy = SelectionPolicy(
            selector,
            min_watermark_words=min_watermark_words,
            max_watermark_words=max_watermark_words,
        )
        self.selection_config_sha256 = self.selection_policy.sha256
        self.selector_resource_sha256 = _selector_resource_sha256(selector)
        self.presence_resource_sha256 = _presence_resource_sha256(presence)
        self.detector_config = {
            "version": DETECTOR_CONFIG_VERSION,
            "selection_config_sha256": self.selection_config_sha256,
            "detector_profile": detector_profile,
            "presence_mode": presence_mode,
            "presence_fingerprint": dict(presence.fingerprint),
            "similarity_threshold": presence.fingerprint.get(
                "similarity_threshold"
            ),
        }
        self.detector_config_sha256 = sha256_json(self.detector_config)

    @property
    def config_consistent(self) -> bool:
        return bool(self.selector.config_consistent)

    @property
    def eligible_for_aggregate(self) -> bool:
        return bool(self.selector.eligible_for_aggregate and self.config_consistent)

    def score_text(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ConfigurationError("Detector input text must be non-empty")
        try:
            expected_words = self.selection_policy.select_words(text)
        except SelectionError as exc:
            reason = (
                "k_exceeds_vocabulary"
                if "exceeds candidate vocabulary" in str(exc)
                else "selection_failed"
            )
            return {
                "status": "failed",
                "expected_words": [],
                "present_words": [],
                "missing_words": [],
                "watermark_score": 0.0,
                "failure_reason": reason,
                "failure_detail": str(exc),
                "token_form_count": 0,
            }
        if not expected_words:
            return {
                "status": "failed",
                "expected_words": [],
                "present_words": [],
                "missing_words": [],
                "watermark_score": 0.0,
                "failure_reason": "k_zero",
                "token_form_count": 0,
            }
        result = self.presence.score(text, expected_words)
        return {
            "status": "completed",
            "expected_words": expected_words,
            "present_words": list(result.present_words),
            "missing_words": list(result.missing_words),
            "watermark_score": float(result.score),
            "failure_reason": None,
            "token_form_count": int(result.token_form_count),
        }


def _sample_id(record: Mapping[str, Any], id_field: str | None) -> str:
    if id_field and id_field in record:
        value = record[id_field]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
            raise JsonlError(f"Sample ID field {id_field!r} must be a string or integer")
        return str(value)
    return stable_content_id(record)


def _validate_record_contract(
    record: Mapping[str, Any], sample_id: str, detector: BlindPostMarkDetector
) -> None:
    recorded_selection = record.get("selection_config_sha256")
    if (
        recorded_selection is not None
        and recorded_selection != detector.selection_config_sha256
    ):
        raise ResourceMismatchError(
            f"Input id {sample_id!r} selection config differs from detector"
        )
    recorded_resource = record.get("selector_resource_sha256")
    if (
        recorded_resource is not None
        and recorded_resource != detector.selector_resource_sha256
    ):
        raise ResourceMismatchError(
            f"Input id {sample_id!r} selector resource differs from detector"
        )


def _record_metadata(
    record: Mapping[str, Any], detector: BlindPostMarkDetector
) -> dict[str, Any]:
    input_consistent = record.get("config_consistent", True) is True
    input_eligible = record.get("eligible_for_aggregate", True) is True
    config_consistent = detector.config_consistent and input_consistent
    return {
        "implementation_profile": detector.selector.selection_config.get(
            "implementation_profile"
        ),
        "detector_profile": detector.detector_profile,
        "presence_mode": detector.presence_mode,
        "selection_config_sha256": detector.selection_config_sha256,
        "detector_config_sha256": detector.detector_config_sha256,
        "selector_resource_sha256": detector.selector_resource_sha256,
        "presence_resource_sha256": detector.presence_resource_sha256,
        "config_consistent": config_consistent,
        "eligible_for_aggregate": (
            detector.eligible_for_aggregate and config_consistent and input_eligible
        ),
        "paper_method_compatible": False,
        "exact_paper_reproduction": False,
    }


def run_detection_pipeline(
    *,
    input_path: str,
    output_path: str,
    detector: BlindPostMarkDetector,
    text_field: str = "text",
    id_field: str | None = "id",
    manifest_path: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    if not text_field:
        raise ConfigurationError("text_field cannot be empty")
    output = Path(output_path)
    output_manifest = (
        Path(manifest_path)
        if manifest_path
        else output.with_name(output.name + ".manifest.json")
    )
    if (output.exists() or output_manifest.exists()) and not overwrite:
        raise ConfigurationError("Detection output already exists; use --overwrite")

    records = load_jsonl(input_path, limit=limit)
    seen_ids: set[str] = set()
    outputs: list[dict[str, Any]] = []
    failed = 0
    for record in records:
        sample_id = _sample_id(record, id_field)
        if sample_id in seen_ids:
            raise DuplicateIdError(f"Duplicate detection input ID {sample_id!r}")
        seen_ids.add(sample_id)
        text = record.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise JsonlError(f"Input id {sample_id!r} has no non-empty {text_field!r} string")
        _validate_record_contract(record, sample_id, detector)

        scored = detector.score_text(text)
        failed += scored["status"] == "failed"
        outputs.append(
            {
                "id": sample_id,
                "text": text,
                **scored,
                "input_sha256": sha256_json(record),
                **_record_metadata(record, detector),
            }
        )

    atomic_write_jsonl(output, outputs)
    atomic_write_json(
        output_manifest,
        {
            "version": DETECTOR_CONFIG_VERSION,
            "detector_config_sha256": detector.detector_config_sha256,
            "selection_config_sha256": detector.selection_config_sha256,
            "selector_resource_sha256": detector.selector_resource_sha256,
            "presence_resource_sha256": detector.presence_resource_sha256,
            "detector_config": detector.detector_config,
            "config_consistent": detector.config_consistent,
            "eligible_for_aggregate": detector.eligible_for_aggregate,
            "paper_method_compatible": False,
            "exact_paper_reproduction": False,
        },
    )
    return {"input": len(records), "written": len(outputs), "failed": failed}


def run_paired_detection_pipeline(
    *,
    input_path: str,
    calibration_path: str,
    output_path: str,
    detector: BlindPostMarkDetector,
    negative_field: str = "text1",
    positive_field: str = "text2",
    calibration_text_field: str = "text",
    id_field: str | None = "id",
    manifest_path: str | None = None,
    target_fpr: float = 0.01,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    formal_min_negatives: int = 1000,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not negative_field or not positive_field or negative_field == positive_field:
        raise ConfigurationError("Paired negative and positive fields must be distinct")
    if not calibration_text_field:
        raise ConfigurationError("calibration_text_field cannot be empty")
    output = Path(output_path)
    output_manifest = (
        Path(manifest_path)
        if manifest_path
        else output.with_name(output.name + ".manifest.json")
    )
    if (output.exists() or output_manifest.exists()) and not overwrite:
        raise ConfigurationError("Detection output already exists; use --overwrite")

    held_records = load_jsonl(input_path, limit=limit)
    calibration_records = load_jsonl(calibration_path)
    held_ids: set[str] = set()
    calibration_ids: set[str] = set()
    indexed_held: list[tuple[str, dict[str, Any]]] = []
    indexed_calibration: list[tuple[str, dict[str, Any]]] = []
    for record in held_records:
        sample_id = _sample_id(record, id_field)
        if sample_id in held_ids:
            raise DuplicateIdError(f"Duplicate held-out ID {sample_id!r}")
        held_ids.add(sample_id)
        indexed_held.append((sample_id, record))
    for record in calibration_records:
        sample_id = _sample_id(record, id_field)
        if sample_id in calibration_ids:
            raise DuplicateIdError(f"Duplicate calibration ID {sample_id!r}")
        calibration_ids.add(sample_id)
        indexed_calibration.append((sample_id, record))
    overlap = held_ids & calibration_ids
    if overlap:
        raise ConfigurationError(
            f"Calibration and held-out IDs overlap: {sorted(overlap)[:3]}"
        )

    output_records: list[dict[str, Any]] = []
    score1_values: list[float] = []
    score2_values: list[float] = []
    failed = 0
    for sample_id, record in indexed_held:
        _validate_record_contract(record, sample_id, detector)
        negative_text = record.get(negative_field)
        positive_text = record.get(positive_field)
        if not isinstance(negative_text, str) or not negative_text.strip():
            raise JsonlError(
                f"Held-out id {sample_id!r} has no non-empty {negative_field!r} string"
            )
        if not isinstance(positive_text, str) or not positive_text.strip():
            raise JsonlError(
                f"Held-out id {sample_id!r} has no non-empty {positive_field!r} string"
            )
        negative = detector.score_text(negative_text)
        positive = detector.score_text(positive_text)
        score1_values.append(negative["watermark_score"])
        score2_values.append(positive["watermark_score"])
        sample_failed = negative["status"] == "failed" or positive["status"] == "failed"
        failed += sample_failed
        output_records.append(
            {
                "id": sample_id,
                "status": "failed" if sample_failed else "completed",
                "text1": negative_text,
                "text2": positive_text,
                "score1": negative["watermark_score"],
                "score2": positive["watermark_score"],
                "expected_words1": negative["expected_words"],
                "expected_words2": positive["expected_words"],
                "negative_detection": negative,
                "positive_detection": positive,
                "input_sha256": sha256_json(record),
                **_record_metadata(record, detector),
            }
        )

    calibration_scores: list[float] = []
    for sample_id, record in indexed_calibration:
        text = record.get(calibration_text_field)
        if not isinstance(text, str) or not text.strip():
            raise JsonlError(
                f"Calibration id {sample_id!r} has no non-empty "
                f"{calibration_text_field!r} string"
            )
        calibration_scores.append(detector.score_text(text)["watermark_score"])

    metrics, evaluation_config = evaluate_paired_scores(
        sample_ids=[sample_id for sample_id, _ in indexed_held],
        negative_scores=score1_values,
        positive_scores=score2_values,
        calibration_ids=[sample_id for sample_id, _ in indexed_calibration],
        calibration_negative_scores=calibration_scores,
        detector_config_sha256=detector.detector_config_sha256,
        target_fpr=target_fpr,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        formal_min_negatives=formal_min_negatives,
        calibration_content_sha256=sha256_json(
            sorted(
                (sample_id, sha256_json(record))
                for sample_id, record in indexed_calibration
            )
        ),
        heldout_content_sha256=sha256_json(
            sorted(
                (sample_id, sha256_json(record))
                for sample_id, record in indexed_held
            )
        ),
    )
    for record in output_records:
        record["evaluation_config_sha256"] = metrics["evaluation_config_sha256"]
    atomic_write_jsonl(output, output_records)
    atomic_write_json(
        output_manifest,
        {
            "version": DETECTOR_CONFIG_VERSION,
            "detector_config_sha256": detector.detector_config_sha256,
            "selection_config_sha256": detector.selection_config_sha256,
            "selector_resource_sha256": detector.selector_resource_sha256,
            "presence_resource_sha256": detector.presence_resource_sha256,
            "detector_config": detector.detector_config,
            "evaluation_config": evaluation_config,
            "metrics": metrics,
            "config_consistent": detector.config_consistent,
            "eligible_for_aggregate": detector.eligible_for_aggregate,
            "paper_method_compatible": False,
            "exact_paper_reproduction": False,
        },
    )
    return {
        "input": len(held_records),
        "written": len(output_records),
        "failed": failed,
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect PostMark-Local watermarks using local keyed resources."
    )
    parser.add_argument("--config", default="configs/postmark_portable.json")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--table_path")
    parser.add_argument("--embedder_path")
    parser.add_argument("--embedder_tokenizer_path")
    parser.add_argument("--implementation_profile", choices=("compat", "portable"))
    parser.add_argument("--detector_profile", choices=("portable",), default="portable")
    parser.add_argument(
        "--selection_mode",
        choices=("official_two_stage", "anchor_only", "direct_word"),
    )
    parser.add_argument(
        "--presence_mode", choices=("exact_lemma", "nomic_fuzzy")
    )
    parser.add_argument("--spacy_model")
    parser.add_argument("--similarity_threshold", type=float)
    parser.add_argument("--max_content_tokens", type=int)
    parser.add_argument("--min_token_length", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--min_watermark_words", type=int)
    parser.add_argument("--max_watermark_words", type=int)
    parser.add_argument("--manifest_path")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--negative_field", default="text1")
    parser.add_argument("--positive_field", default="text2")
    parser.add_argument("--calibration_path")
    parser.add_argument("--calibration_text_field", default="text")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--formal_min_negatives", type=int, default=1000)
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow_resource_mismatch", action="store_true")
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_offline_environment()
    from .nomic_embedder import NomicPostMarkEmbedder
    from .presence import ExactLemmaPresence, NomicFuzzyPresence

    install_network_guard_from_environment()
    if not args.local_files_only:
        raise ConfigurationError("PostMark-Local requires --local_files_only")
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = PostMarkConfig.load(config_path)
    paths = config.paths.resolved(project_root)
    selector = NomicPostMarkEmbedder(
        args.embedder_path or str(paths["embedder"]),
        args.table_path or str(paths["nomic_table"]),
        tokenizer_path=args.embedder_tokenizer_path or str(paths["embedder_tokenizer"]),
        implementation_profile=args.implementation_profile or config.implementation_profile,
        selection_mode=args.selection_mode or config.selection_mode,
        ratio=args.ratio if args.ratio is not None else config.selection.ratio,
        max_length=config.embedding.max_length,
        local_files_only=True,
        allow_resource_mismatch=args.allow_resource_mismatch,
        allow_config_mismatch=args.allow_config_mismatch,
    )
    presence_mode = args.presence_mode or config.detector.presence_mode
    similarity_threshold = (
        args.similarity_threshold
        if args.similarity_threshold is not None
        else config.detector.similarity_threshold
    )
    max_content_tokens = (
        args.max_content_tokens
        if args.max_content_tokens is not None
        else config.detector.max_content_tokens
    )
    min_token_length = (
        args.min_token_length
        if args.min_token_length is not None
        else config.detector.min_token_length
    )
    exact_presence = ExactLemmaPresence(
        args.spacy_model or config.detector.spacy_model
    )
    if presence_mode == "nomic_fuzzy":
        encoder_contract = {
            key: selector.selection_config[key]
            for key in (
                "embedder_fingerprint",
                "tokenizer_fingerprint",
                "pooling",
                "normalization",
                "max_length",
                "task_prefix",
            )
        }
        presence = NomicFuzzyPresence(
            exact_presence,
            selector,
            encoder_fingerprint=encoder_contract,
            similarity_threshold=similarity_threshold,
            max_content_tokens=max_content_tokens,
            min_token_length=min_token_length,
        )
    else:
        presence = exact_presence
    detector = BlindPostMarkDetector(
        selector,
        presence,
        detector_profile=args.detector_profile,
        presence_mode=presence_mode,
        min_watermark_words=args.min_watermark_words,
        max_watermark_words=args.max_watermark_words,
    )
    if args.paired:
        if not args.calibration_path:
            raise ConfigurationError("--paired requires --calibration_path")
        result = run_paired_detection_pipeline(
            input_path=args.input_path,
            calibration_path=args.calibration_path,
            output_path=args.output_path,
            detector=detector,
            negative_field=args.negative_field,
            positive_field=args.positive_field,
            calibration_text_field=args.calibration_text_field,
            id_field=args.id_field,
            manifest_path=args.manifest_path,
            target_fpr=args.target_fpr,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            formal_min_negatives=args.formal_min_negatives,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    else:
        if args.calibration_path:
            raise ConfigurationError("--calibration_path requires --paired")
        result = run_detection_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            detector=detector,
            text_field=args.text_field,
            id_field=args.id_field,
            manifest_path=args.manifest_path,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    print(
        f"processed={result['input']} written={result['written']} failed={result['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
