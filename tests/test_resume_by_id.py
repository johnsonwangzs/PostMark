from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from postmark.common import ConfigurationError, DuplicateIdError, atomic_write_jsonl, load_jsonl
from postmark.watermark import PostMarkWatermarker, run_watermark_pipeline
from tests.watermark_fixtures import EchoInserter, FakeSelector


class ResumeByIdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prompt = self.root / "insert.txt"
        self.prompt.write_text("TEXT={}\nWORDS={}", encoding="utf-8")
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def watermarker(self) -> PostMarkWatermarker:
        return PostMarkWatermarker(
            EchoInserter(), FakeSelector(), prompt_path=str(self.prompt), max_new_tokens=20
        )

    def write_input(self, records) -> None:
        self.input.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def run_pipeline(self, watermarker=None):
        return run_watermark_pipeline(
            input_path=str(self.input),
            output_path=str(self.output),
            watermarker=watermarker or self.watermarker(),
            text_field="text",
        )

    def test_reorder_and_insert_resume_by_id(self) -> None:
        self.write_input([{"id": "a", "text": "A text"}, {"id": "b", "text": "B text"}])
        self.run_pipeline()
        self.write_input(
            [
                {"id": "c", "text": "C text"},
                {"id": "b", "text": "B text"},
                {"id": "a", "text": "A text"},
            ]
        )
        result = self.run_pipeline()
        self.assertEqual(result, {"input": 3, "written": 1, "skipped": 2})
        self.assertEqual({record["id"] for record in load_jsonl(self.output)}, {"a", "b", "c"})

    def test_changed_input_for_existing_id_is_a_hard_conflict(self) -> None:
        self.write_input([{"id": "a", "text": "first"}])
        self.run_pipeline()
        self.write_input([{"id": "a", "text": "changed"}])
        with self.assertRaisesRegex(ConfigurationError, "input_sha256"):
            self.run_pipeline()

    def test_duplicate_input_id_is_rejected(self) -> None:
        self.write_input([{"id": "a", "text": "first"}, {"id": "a", "text": "second"}])
        with self.assertRaises(DuplicateIdError):
            self.run_pipeline()

    def test_existing_output_without_manifest_is_rejected(self) -> None:
        self.write_input([{"id": "a", "text": "first"}])
        self.output.write_text('{"id":"a"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "no run manifest"):
            self.run_pipeline()

    def test_nonterminal_record_is_replaced_and_truncated_tail_is_recovered(self) -> None:
        self.write_input([{"id": "a", "text": "A text"}])
        self.run_pipeline()
        record = load_jsonl(self.output)[0]
        record["status"] = "running"
        atomic_write_jsonl(self.output, [record])
        with self.output.open("ab") as handle:
            handle.write(b'{"id":"broken"')

        result = self.run_pipeline()
        self.assertEqual(result, {"input": 1, "written": 1, "skipped": 0})
        records = load_jsonl(self.output)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "completed")
        self.assertTrue(list(self.root.glob("output.jsonl.corrupt.bak*")))


if __name__ == "__main__":
    unittest.main()
