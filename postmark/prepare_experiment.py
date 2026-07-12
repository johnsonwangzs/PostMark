"""Freeze deterministic pilot, held-out, and calibration JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    ConfigurationError,
    DuplicateIdError,
    JsonlError,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    iter_jsonl,
    sha256_file,
    sha256_json,
)


DATASET_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    text: str
    token_count: int
    text_sha256: str

    def output_record(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "source_token_count": self.token_count,
            "text": self.text,
        }


def _require_text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise JsonlError(f"{context} field {field!r} must be a non-empty string")
    return value


def _load_source(
    path: str | Path,
    *,
    id_field: str,
    text_field: str,
    token_trace_field: str,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for line_number, record in enumerate(iter_jsonl(path), start=1):
        context = f"{path}:{line_number}"
        sample_id = _require_text(record, id_field, context)
        if sample_id in seen_ids:
            raise DuplicateIdError(f"Duplicate sample ID {sample_id!r} in {path}")
        text = _require_text(record, text_field, context)
        text_sha256 = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if text_sha256 in seen_texts:
            raise JsonlError(f"Duplicate normalized text in {path} at ID {sample_id!r}")
        trace = record.get(token_trace_field)
        if not isinstance(trace, list):
            raise JsonlError(
                f"{context} field {token_trace_field!r} must be a token trace list"
            )
        seen_ids.add(sample_id)
        seen_texts.add(text_sha256)
        records.append(SourceRecord(sample_id, text, len(trace), text_sha256))
    return records


def _rank(seed: int, namespace: str, sample_id: str) -> bytes:
    return hashlib.sha256(
        canonical_json_bytes(
            {"namespace": namespace, "sample_id": sample_id, "seed": seed}
        )
    ).digest()


def _select(
    records: Sequence[SourceRecord],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> list[SourceRecord]:
    if count < 1:
        raise ConfigurationError(f"{namespace} count must be positive")
    if len(records) < count:
        raise ConfigurationError(
            f"{namespace} requires {count} eligible records, found {len(records)}"
        )
    return sorted(
        records,
        key=lambda record: (_rank(seed, namespace, record.sample_id), record.sample_id),
    )[:count]


def _split_manifest(path: Path, records: Sequence[SourceRecord]) -> dict[str, Any]:
    token_counts = [record.token_count for record in records]
    return {
        "path": str(path.resolve()),
        "count": len(records),
        "ids_sha256": sha256_json([record.sample_id for record in records]),
        "content_sha256": sha256_file(path),
        "token_count": {
            "min": min(token_counts),
            "median": statistics.median(token_counts),
            "max": max(token_counts),
        },
    }


def prepare_experiment_splits(
    *,
    test_source_path: str | Path,
    calibration_source_path: str | Path,
    output_dir: str | Path,
    seed: int = 1618,
    test_count: int = 200,
    pilot_count: int = 30,
    calibration_count: int = 1000,
    detector_dev_count: int = 200,
    min_tokens: int = 256,
    max_tokens: int = 512,
    id_field: str = "record_id",
    text_field: str = "response",
    token_trace_field: str = "trace",
    target_fpr: float = 0.01,
) -> dict[str, Any]:
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ConfigurationError("Token bounds must satisfy 1 <= min_tokens <= max_tokens")
    if not 0 < target_fpr < 1:
        raise ConfigurationError("target_fpr must be strictly between 0 and 1")

    test_source = Path(test_source_path).resolve()
    calibration_source = Path(calibration_source_path).resolve()
    run_records = _load_source(
        test_source,
        id_field=id_field,
        text_field=text_field,
        token_trace_field=token_trace_field,
    )
    calibration_records = _load_source(
        calibration_source,
        id_field=id_field,
        text_field=text_field,
        token_trace_field=token_trace_field,
    )

    run_ids = {record.sample_id for record in run_records}
    calibration_ids = {record.sample_id for record in calibration_records}
    overlapping_ids = run_ids & calibration_ids
    if overlapping_ids:
        raise JsonlError(
            f"Test and calibration sources overlap by {len(overlapping_ids)} IDs"
        )
    run_texts = {record.text_sha256 for record in run_records}
    calibration_texts = {record.text_sha256 for record in calibration_records}
    overlapping_texts = run_texts & calibration_texts
    if overlapping_texts:
        raise JsonlError(
            f"Test and calibration sources overlap by {len(overlapping_texts)} texts"
        )

    def eligible(records: Sequence[SourceRecord]) -> list[SourceRecord]:
        return [
            record
            for record in records
            if min_tokens <= record.token_count <= max_tokens
        ]

    eligible_run = eligible(run_records)
    eligible_calibration = eligible(calibration_records)
    test = _select(
        eligible_run,
        count=test_count,
        seed=seed,
        namespace="formal_test",
    )
    test_ids = {record.sample_id for record in test}
    pilot = _select(
        [record for record in eligible_run if record.sample_id not in test_ids],
        count=pilot_count,
        seed=seed,
        namespace="pilot",
    )
    calibration = _select(
        eligible_calibration,
        count=calibration_count,
        seed=seed,
        namespace="calibration",
    )
    calibration_ids = {record.sample_id for record in calibration}
    detector_dev = _select(
        [
            record
            for record in eligible_calibration
            if record.sample_id not in calibration_ids
        ],
        count=detector_dev_count,
        seed=seed,
        namespace="detector_dev",
    )

    output_root = Path(output_dir).resolve()
    paths = {
        "pilot": output_root / "pilot.jsonl",
        "test": output_root / f"test_{test_count}.jsonl",
        "calibration": output_root / f"calibration_{calibration_count}.jsonl",
        "detector_dev": output_root / f"detector_dev_{detector_dev_count}.jsonl",
    }
    for name, records in (
        ("pilot", pilot),
        ("test", test),
        ("calibration", calibration),
        ("detector_dev", detector_dev),
    ):
        atomic_write_jsonl(paths[name], (record.output_record() for record in records))

    config = {
        "seed": seed,
        "test_count": test_count,
        "pilot_count": pilot_count,
        "calibration_count": calibration_count,
        "detector_dev_count": detector_dev_count,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "id_field": id_field,
        "text_field": text_field,
        "token_trace_field": token_trace_field,
        "target_fpr": target_fpr,
        "selection_algorithm": "sha256_id_rank_v1",
        "split_order": ["formal_test", "pilot", "calibration", "detector_dev"],
    }
    manifest: dict[str, Any] = {
        "schema_version": DATASET_MANIFEST_VERSION,
        "experiment": {
            "name": f"postmark_local_{test_count}_pairs",
            "primary_detector": "nomic_fuzzy",
            "secondary_detector": "exact_lemma",
            "paragram_in_scope": False,
            "held_out_pairs": test_count,
            "target_fpr": target_fpr,
        },
        "preparation": config,
        "preparation_config_sha256": sha256_json(config),
        "sources": {
            "test_candidates": {
                "path": str(test_source),
                "sha256": sha256_file(test_source),
                "total_count": len(run_records),
                "eligible_count": len(eligible_run),
                "length_filtered_count": len(run_records) - len(eligible_run),
            },
            "calibration_candidates": {
                "path": str(calibration_source),
                "sha256": sha256_file(calibration_source),
                "total_count": len(calibration_records),
                "eligible_count": len(eligible_calibration),
                "length_filtered_count": len(calibration_records)
                - len(eligible_calibration),
            },
        },
        "leakage_checks": {
            "source_id_overlap": 0,
            "source_exact_text_overlap": 0,
            "pilot_test_id_overlap": 0,
            "detector_dev_calibration_id_overlap": 0,
        },
        "splits": {
            name: _split_manifest(paths[name], records)
            for name, records in (
                ("pilot", pilot),
                ("test", test),
                ("calibration", calibration),
                ("detector_dev", detector_dev),
            )
        },
    }
    manifest["dataset_manifest_sha256"] = sha256_json(manifest)
    atomic_write_json(output_root / "dataset_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic PostMark pilot, test, and calibration data."
    )
    parser.add_argument("--test_source_path", required=True)
    parser.add_argument("--calibration_source_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=1618)
    parser.add_argument("--test_count", type=int, default=200)
    parser.add_argument("--pilot_count", type=int, default=30)
    parser.add_argument("--calibration_count", type=int, default=1000)
    parser.add_argument("--detector_dev_count", type=int, default=200)
    parser.add_argument("--min_tokens", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--id_field", default="record_id")
    parser.add_argument("--text_field", default="response")
    parser.add_argument("--token_trace_field", default="trace")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_experiment_splits(**vars(args))
    splits = manifest["splits"]
    print(
        "Prepared "
        f"pilot={splits['pilot']['count']}, "
        f"test={splits['test']['count']}, "
        f"calibration={splits['calibration']['count']}, "
        f"detector_dev={splits['detector_dev']['count']} "
        f"(manifest_sha256={manifest['dataset_manifest_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
