from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from postmark.common import (
    ConfigurationError,
    DuplicateIdError,
    JsonlError,
    load_json_object,
    load_jsonl,
)
from postmark.quality import (
    NomicSemanticEvaluator,
    aggregate_quality_metrics,
    sample_quality_metrics,
    write_quality_report,
)


def _completed(sample_id: str = "a") -> dict:
    return {
        "id": sample_id,
        "status": "completed",
        "text1": "one two",
        "list1": ["alpha", "beta"],
        "text2": "one alpha two",
        "list2": ["alpha"],
        "diagnostics": {
            "list_overlap": 0.5,
            "requested_word_presence": 0.5,
            "length_delta_words": 1,
            "groups": [
                {
                    "best_presence": 0.5,
                    "threshold_met": True,
                    "max_attempt_exhausted": False,
                }
            ],
            "insertion_success": True,
            "max_attempt_exhausted": False,
            "empty_output": False,
            "embedding_input_truncated": False,
            "generation_input_truncated": False,
            "generation_output_truncated": True,
        },
        "selection_config_sha256": "selection",
        "run_config_sha256": "run",
        "eligible_for_aggregate": True,
        "task1": 0.7,
        "task2": 0.6,
    }


def _failed(sample_id: str = "b") -> dict:
    return {
        "id": sample_id,
        "status": "failed",
        "text1": "short text",
        "list1": [],
        "text2": "short text",
        "list2": [],
        "diagnostics": {
            "list_overlap": 1.0,
            "requested_word_presence": 0.0,
            "length_delta_words": 0,
            "groups": [],
            "insertion_success": False,
            "max_attempt_exhausted": False,
            "empty_output": False,
            "embedding_input_truncated": True,
            "generation_input_truncated": False,
            "generation_output_truncated": False,
            "failure_reason": "k_zero",
        },
        "selection_config_sha256": "selection",
        "run_config_sha256": "run",
        "eligible_for_aggregate": True,
        "task1": 0.3,
        "task2": 0.3,
    }


class SampleQualityTests(unittest.TestCase):
    def test_recomputes_presence_overlap_lengths_and_task_delta(self) -> None:
        metrics = sample_quality_metrics(
            _completed(), semantic_similarity=0.8, task_score1=0.7, task_score2=0.6
        )
        self.assertEqual(metrics["requested_word_presence"], 0.5)
        self.assertEqual(metrics["list_overlap"], 0.5)
        self.assertEqual(metrics["length_delta_words"], 1)
        self.assertEqual(metrics["relative_length_delta"], 0.5)
        self.assertAlmostEqual(metrics["task_score_delta"], -0.1)
        self.assertTrue(metrics["insertion_success"])
        self.assertTrue(metrics["generation_output_truncated"])

    def test_empty_text_has_explicit_zero_denominator(self) -> None:
        record = _failed()
        record.update({"text1": "", "text2": ""})
        metrics = sample_quality_metrics(record)
        self.assertTrue(metrics["zero_length_denominator"])
        self.assertIsNone(metrics["relative_length_delta"])
        self.assertEqual(metrics["text1_words"], 0)

    def test_inconsistent_recorded_diagnostics_are_rejected(self) -> None:
        record = _completed()
        record["diagnostics"]["requested_word_presence"] = 1.0
        with self.assertRaisesRegex(ConfigurationError, "requested_word_presence"):
            sample_quality_metrics(record)


class AggregateQualityTests(unittest.TestCase):
    def records(self) -> list[dict]:
        ineligible = _completed("c")
        ineligible["eligible_for_aggregate"] = False
        ineligible["selection_config_sha256"] = "other-selection"
        ineligible["run_config_sha256"] = "other-run"
        return [_completed(), _failed(), ineligible]

    def test_coverage_failures_rates_distributions_and_fingerprints(self) -> None:
        report, samples = aggregate_quality_metrics(
            self.records(),
            semantic_scores={"a": 0.8, "b": 0.4},
            semantic_evaluator_fingerprint={
                "evaluator_type": "nomic_proxy",
                "sha256": "semantic",
            },
            task_score1_field="task1",
            task_score2_field="task2",
            task_evaluator_fingerprint={"name": "fixture-task", "sha256": "task"},
        )
        self.assertEqual(report["coverage"]["total_input"], 3)
        self.assertEqual(report["coverage"]["eligible"], 2)
        self.assertEqual(report["coverage"]["ineligible"], 1)
        self.assertEqual(
            report["coverage"]["status_counts"], {"completed": 1, "failed": 1}
        )
        self.assertEqual(report["coverage"]["failure_class_counts"], {"k_zero": 1})
        self.assertEqual(report["rates"]["insertion_success"], 0.5)
        self.assertEqual(report["rates"]["embedding_input_truncated"], 0.5)
        self.assertEqual(report["rates"]["generation_output_truncated"], 0.5)
        self.assertAlmostEqual(
            report["distributions"]["semantic_similarity"]["mean"], 0.6
        )
        self.assertEqual(report["distributions"]["length_delta_words"]["p95"], 0.95)
        self.assertAlmostEqual(report["distributions"]["task_score_delta"]["mean"], -0.05)
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(report["quality_config_sha256"]), 64)
        self.assertTrue(report["clean_condition_only"])

    def test_nan_duplicate_and_mixed_eligible_configs_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            aggregate_quality_metrics(
                [_completed()],
                semantic_scores={"a": math.nan},
                semantic_evaluator_fingerprint={"sha256": "semantic"},
            )
        with self.assertRaises(DuplicateIdError):
            aggregate_quality_metrics([_completed(), _completed()])
        mixed = [_completed(), _failed()]
        mixed[1]["run_config_sha256"] = "different"
        with self.assertRaisesRegex(ConfigurationError, "mix"):
            aggregate_quality_metrics(mixed)

        invalid_eligible = _completed()
        invalid_eligible["eligible_for_aggregate"] = "yes"
        with self.assertRaisesRegex(JsonlError, "eligible_for_aggregate"):
            aggregate_quality_metrics([invalid_eligible])

    def test_input_reordering_preserves_report_hash_and_aggregates(self) -> None:
        records = [_completed(), _failed()]
        first, _ = aggregate_quality_metrics(records)
        second, _ = aggregate_quality_metrics(list(reversed(records)))
        self.assertEqual(first, second)

    def test_report_and_sample_jsonl_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_path = root / "quality.json"
            sample_path = root / "samples.jsonl"
            from postmark.common import atomic_write_jsonl

            atomic_write_jsonl(input_path, [_completed(), _failed()])
            report = write_quality_report(
                input_path=str(input_path),
                output_path=str(output_path),
                sample_output_path=str(sample_path),
            )
            self.assertEqual(
                load_json_object(output_path)["quality_config_sha256"],
                report["quality_config_sha256"],
            )
            self.assertEqual(len(load_jsonl(sample_path)), 2)
            with self.assertRaisesRegex(ConfigurationError, "already exists"):
                write_quality_report(
                    input_path=str(input_path), output_path=str(output_path)
                )


class _SemanticEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        self.calls.append(texts.copy())
        vectors = {
            "one two": [1.0, 0.0],
            "one alpha two": [0.8, 0.6],
        }
        return torch.tensor([vectors[text] for text in texts])


class NomicSemanticEvaluatorTests(unittest.TestCase):
    def test_scores_eligible_pairs_and_labels_proxy_fingerprint(self) -> None:
        encoder = _SemanticEncoder()
        evaluator = NomicSemanticEvaluator(
            encoder, encoder_contract={"sha256": "encoder-fixture"}
        )
        empty = _failed("empty")
        empty.update({"text1": "", "text2": ""})
        ineligible = _completed("skip")
        ineligible["eligible_for_aggregate"] = False
        scores = evaluator.score_records([_completed(), empty, ineligible])

        self.assertAlmostEqual(scores["a"], 0.8)
        self.assertEqual(scores["empty"], 0.0)
        self.assertNotIn("skip", scores)
        self.assertEqual(evaluator.fingerprint["evaluator_type"], "nomic_proxy")
        self.assertEqual(len(evaluator.fingerprint["sha256"]), 64)
        self.assertEqual(len(encoder.calls), 2)


if __name__ == "__main__":
    unittest.main()
