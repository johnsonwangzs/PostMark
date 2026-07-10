import json
import random
import tempfile
import unittest
from pathlib import Path

from postmark.common import (
    ConfigurationError,
    DuplicateIdError,
    JsonlError,
    append_jsonl_record,
    canonical_json_dumps,
    config_sha256,
    derive_sample_seed,
    index_records_by_id,
    load_json_object,
    load_jsonl,
    recover_truncated_jsonl_tail,
    set_global_seed,
    stable_content_id,
    stable_word_count,
)


class CanonicalJsonTests(unittest.TestCase):
    def test_hash_is_independent_of_mapping_order(self):
        left = {"b": [2, 3], "a": 1}
        right = {"a": 1, "b": [2, 3]}
        self.assertEqual(canonical_json_dumps(left), canonical_json_dumps(right))
        self.assertEqual(config_sha256(left), config_sha256(right))

    def test_non_finite_values_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            canonical_json_dumps({"score": float("nan")})

    def test_config_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_json_object(path)


class DeterminismTests(unittest.TestCase):
    def test_sample_seed_is_stable_and_attempt_specific(self):
        first = derive_sample_seed(42, "sample-a")
        self.assertEqual(first, derive_sample_seed(42, "sample-a"))
        self.assertNotEqual(first, derive_sample_seed(42, "sample-a", attempt=1))
        self.assertNotEqual(first, derive_sample_seed(42, "sample-b"))

    def test_global_seed_controls_python_random(self):
        set_global_seed(9)
        first = random.random()
        set_global_seed(9)
        self.assertEqual(first, random.random())

    def test_stable_helpers(self):
        self.assertEqual(stable_word_count("  one\ttwo\nthree "), 3)
        self.assertEqual(
            stable_content_id({"text": "same"}),
            stable_content_id({"text": "same"}),
        )


class JsonlTests(unittest.TestCase):
    def test_append_load_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            append_jsonl_record(path, {"text": "first", "id": "a"})
            append_jsonl_record(path, {"id": 2, "text": "second"})
            records = load_jsonl(path)
            self.assertEqual([record["id"] for record in records], ["a", 2])
            self.assertEqual(set(index_records_by_id(records)), {"a", "2"})
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(DuplicateIdError):
            index_records_by_id([{"id": "x"}, {"id": "x"}])

    def test_only_truncated_tail_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_bytes(b'{"id":"ok"}\n{"id":')
            backup = recover_truncated_jsonl_tail(path)
            self.assertIsNotNone(backup)
            self.assertEqual(load_jsonl(path), [{"id": "ok"}])
            self.assertEqual(backup.read_bytes(), b'{"id":"ok"}\n{"id":')

    def test_newline_terminated_corruption_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"id":"ok"}\nnot-json\n{"id":"later"}\n', encoding="utf-8")
            with self.assertRaises(JsonlError):
                recover_truncated_jsonl_tail(path)
            self.assertEqual(len(list(Path(directory).glob("*.corrupt.bak*"))), 0)

    def test_blank_lines_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"id":"ok"}\n\n', encoding="utf-8")
            with self.assertRaises(JsonlError):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
