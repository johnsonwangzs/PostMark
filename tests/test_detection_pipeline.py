from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from postmark.common import ConfigurationError, DuplicateIdError, ResourceMismatchError, load_jsonl
from postmark.detect import (
    BlindPostMarkDetector,
    run_detection_pipeline,
    run_paired_detection_pipeline,
)
from postmark.presence import PresenceResult


class _Selector:
    selection_config_sha256 = "base-selector-hash"
    selection_config = {
        "implementation_profile": "compat",
        "selection_mode": "official_two_stage",
    }
    table_manifest = SimpleNamespace(content_sha256="selector-resource")
    config_consistent = True
    eligible_for_aggregate = True

    def __init__(self, *, k: int = 2) -> None:
        self.k = k
        self.calls: list[tuple[str, int | None]] = []

    def word_count_to_k(self, text: str) -> int:
        return self.k

    def select_words(self, text: str, *, top_k: int | None = None) -> list[str]:
        self.calls.append((text, top_k))
        return ["alpha", "running"][:top_k]


class _Presence:
    fingerprint_sha256 = "presence-config"
    fingerprint = {
        "presence_mode": "exact_lemma",
        "spacy_model": {"fingerprint": {"sha256": "spacy-resource"}},
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, text: str, expected_words) -> PresenceResult:
        words = list(expected_words)
        self.calls.append((text, words))
        present = [word for word in words if word == "alpha"]
        missing = [word for word in words if word not in present]
        return PresenceResult(
            len(present) / len(words) if words else 0.0,
            present,
            missing,
            4,
        )


class BlindDetectionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def detector(self, *, k: int = 2) -> tuple[BlindPostMarkDetector, _Selector, _Presence]:
        selector = _Selector(k=k)
        presence = _Presence()
        return BlindPostMarkDetector(selector, presence), selector, presence

    def write_input(self, records) -> None:
        self.input.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_blind_pipeline_recomputes_words_from_candidate_text(self) -> None:
        detector, selector, presence = self.detector()
        self.write_input(
            [
                {
                    "id": "x",
                    "text2": "candidate alpha text",
                    "list1": ["forged-one"],
                    "list2": ["forged-two"],
                }
            ]
        )
        result = run_detection_pipeline(
            input_path=str(self.input),
            output_path=str(self.output),
            detector=detector,
            text_field="text2",
        )

        self.assertEqual(result, {"input": 1, "written": 1, "failed": 0})
        self.assertEqual(selector.calls, [("candidate alpha text", 2)])
        self.assertEqual(presence.calls, [("candidate alpha text", ["alpha", "running"])])
        record = load_jsonl(self.output)[0]
        self.assertEqual(record["expected_words"], ["alpha", "running"])
        self.assertEqual(record["watermark_score"], 0.5)
        self.assertNotIn("forged-one", record["expected_words"])
        self.assertFalse(record["paper_method_compatible"])
        manifest = json.loads(
            (self.root / "output.jsonl.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["detector_config_sha256"], record["detector_config_sha256"]
        )

    def test_k_zero_is_terminal_zero_without_presence_call(self) -> None:
        detector, selector, presence = self.detector(k=0)
        result = detector.score_text("short text")
        self.assertEqual(selector.calls, [("short text", 0)])
        self.assertEqual(presence.calls, [])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "k_zero")
        self.assertEqual(result["watermark_score"], 0.0)

    def test_recorded_selection_or_resource_mismatch_is_rejected(self) -> None:
        detector, _, _ = self.detector()
        for field in ("selection_config_sha256", "selector_resource_sha256"):
            with self.subTest(field=field):
                self.write_input([{"id": "x", "text": "candidate", field: "wrong"}])
                with self.assertRaises(ResourceMismatchError):
                    run_detection_pipeline(
                        input_path=str(self.input),
                        output_path=str(self.output),
                        detector=detector,
                    )

    def test_duplicate_ids_and_existing_outputs_are_rejected(self) -> None:
        detector, _, _ = self.detector()
        self.write_input([{"id": "x", "text": "one"}, {"id": "x", "text": "two"}])
        with self.assertRaises(DuplicateIdError):
            run_detection_pipeline(
                input_path=str(self.input),
                output_path=str(self.output),
                detector=detector,
            )

        self.write_input([{"id": "x", "text": "one"}])
        self.output.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "already exists"):
            run_detection_pipeline(
                input_path=str(self.input),
                output_path=str(self.output),
                detector=detector,
            )

    def test_paired_pipeline_writes_scores_metrics_and_evaluation_hash(self) -> None:
        detector, _, _ = self.detector()
        self.write_input(
            [
                {"id": "a", "text1": "plain", "text2": "alpha marked"},
                {"id": "b", "text1": "other", "text2": "alpha other"},
            ]
        )
        calibration = self.root / "calibration.jsonl"
        calibration.write_text(
            json.dumps({"id": "n1", "text": "negative one"})
            + "\n"
            + json.dumps({"id": "n2", "text": "negative two"})
            + "\n",
            encoding="utf-8",
        )
        result = run_paired_detection_pipeline(
            input_path=str(self.input),
            calibration_path=str(calibration),
            output_path=str(self.output),
            detector=detector,
            bootstrap_samples=20,
            bootstrap_seed=3,
        )

        self.assertEqual(result["input"], 2)
        self.assertEqual(result["written"], 2)
        self.assertEqual(
            result["metrics"]["metric_status"],
            "diagnostic_insufficient_negatives",
        )
        records = load_jsonl(self.output)
        manifest = json.loads(
            (self.root / "output.jsonl.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(records[0]["score1"], 0.5)
        self.assertEqual(records[0]["score2"], 0.5)
        self.assertEqual(
            records[0]["evaluation_config_sha256"],
            manifest["metrics"]["evaluation_config_sha256"],
        )

    def test_paired_split_overlap_fails_before_scoring(self) -> None:
        detector, selector, presence = self.detector()
        self.write_input([{"id": "same", "text1": "plain", "text2": "marked"}])
        calibration = self.root / "calibration.jsonl"
        calibration.write_text(
            json.dumps({"id": "same", "text": "negative"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ConfigurationError, "overlap"):
            run_paired_detection_pipeline(
                input_path=str(self.input),
                calibration_path=str(calibration),
                output_path=str(self.output),
                detector=detector,
                bootstrap_samples=2,
            )
        self.assertEqual(selector.calls, [])
        self.assertEqual(presence.calls, [])


if __name__ == "__main__":
    unittest.main()
