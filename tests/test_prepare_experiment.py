from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from postmark.common import ConfigurationError, JsonlError, load_jsonl, sha256_file
from postmark.prepare_experiment import prepare_experiment_splits


class PrepareExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_source(self, name: str, records: list[dict]) -> Path:
        path = self.root / name
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def record(sample_id: str, token_count: int) -> dict:
        return {
            "record_id": sample_id,
            "response": f"text for {sample_id}",
            "trace": [{}] * token_count,
        }

    def prepare(self, test_path: Path, calibration_path: Path, output: str):
        return prepare_experiment_splits(
            test_source_path=test_path,
            calibration_source_path=calibration_path,
            output_dir=self.root / output,
            seed=1618,
            test_count=3,
            pilot_count=2,
            calibration_count=4,
            min_tokens=2,
            max_tokens=4,
        )

    def test_splits_are_disjoint_filtered_and_fingerprinted(self) -> None:
        test_path = self.write_source(
            "test.jsonl",
            [self.record(f"test-{index}", index % 4 + 1) for index in range(10)],
        )
        calibration_path = self.write_source(
            "calibration.jsonl",
            [self.record(f"cal-{index}", index % 4 + 1) for index in range(10)],
        )
        manifest = self.prepare(test_path, calibration_path, "first")

        pilot = load_jsonl(self.root / "first" / "pilot.jsonl")
        test = load_jsonl(self.root / "first" / "test_3.jsonl")
        calibration = load_jsonl(self.root / "first" / "calibration_4.jsonl")
        self.assertEqual((len(pilot), len(test), len(calibration)), (2, 3, 4))
        self.assertFalse({row["id"] for row in pilot} & {row["id"] for row in test})
        self.assertTrue(
            all(2 <= row["source_token_count"] <= 4 for row in pilot + test + calibration)
        )
        self.assertFalse(manifest["experiment"]["paragram_in_scope"])
        self.assertEqual(
            manifest["splits"]["test"]["content_sha256"],
            sha256_file(self.root / "first" / "test_3.jsonl"),
        )

    def test_selection_is_independent_of_input_order(self) -> None:
        test_records = [self.record(f"test-{index}", 3) for index in range(8)]
        calibration_records = [self.record(f"cal-{index}", 3) for index in range(8)]
        first_test = self.write_source("test-a.jsonl", test_records)
        first_calibration = self.write_source("cal-a.jsonl", calibration_records)
        self.prepare(first_test, first_calibration, "first")

        second_test = self.write_source("test-b.jsonl", list(reversed(test_records)))
        second_calibration = self.write_source(
            "cal-b.jsonl", list(reversed(calibration_records))
        )
        self.prepare(second_test, second_calibration, "second")

        for filename in ("pilot.jsonl", "test_3.jsonl", "calibration_4.jsonl"):
            self.assertEqual(
                (self.root / "first" / filename).read_bytes(),
                (self.root / "second" / filename).read_bytes(),
            )

    def test_cross_source_text_overlap_is_rejected(self) -> None:
        shared = self.record("test-shared", 3)
        test_path = self.write_source(
            "test.jsonl", [shared] + [self.record(f"test-{i}", 3) for i in range(5)]
        )
        calibration_records = [self.record(f"cal-{i}", 3) for i in range(6)]
        calibration_records[0]["response"] = shared["response"]
        calibration_path = self.write_source("cal.jsonl", calibration_records)
        with self.assertRaises(JsonlError):
            self.prepare(test_path, calibration_path, "out")

    def test_insufficient_eligible_records_is_rejected(self) -> None:
        test_path = self.write_source(
            "test.jsonl", [self.record(f"test-{i}", 1) for i in range(6)]
        )
        calibration_path = self.write_source(
            "cal.jsonl", [self.record(f"cal-{i}", 3) for i in range(6)]
        )
        with self.assertRaises(ConfigurationError):
            self.prepare(test_path, calibration_path, "out")


if __name__ == "__main__":
    unittest.main()
