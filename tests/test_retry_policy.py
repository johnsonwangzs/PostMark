from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from postmark.common import GenerationError
from postmark.watermark import PostMarkWatermarker
from tests.watermark_fixtures import FakeSelector, QueueInserter


class RetryPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.prompt = Path(self.temporary.name) / "insert.txt"
        self.prompt.write_text("TEXT={}\nWORDS={}", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def watermarker(self, responses, **kwargs) -> tuple[PostMarkWatermarker, QueueInserter]:
        inserter = QueueInserter(responses)
        watermarker = PostMarkWatermarker(
            inserter,
            FakeSelector(),
            prompt_path=str(self.prompt),
            group_size=2,
            min_group_presence=1.0,
            max_insert_attempts=2,
            max_new_tokens=20,
            **kwargs,
        )
        return watermarker, inserter

    def test_retry_uses_best_text_and_only_missing_words(self) -> None:
        watermarker, inserter = self.watermarker(
            ["original alpha", "original alpha beta"]
        )
        result = watermarker.insert_watermark("original")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["text2"], "original alpha beta")
        self.assertEqual(len(inserter.prompts), 2)
        self.assertIn("WORDS=alpha, beta", inserter.prompts[0])
        self.assertIn("TEXT=original alpha", inserter.prompts[1])
        self.assertIn("WORDS=beta", inserter.prompts[1])
        attempts = result["diagnostics"]["attempts"]
        self.assertEqual(attempts[0]["stop_reason"], "retry_missing_words")
        self.assertEqual(attempts[1]["stop_reason"], "presence_threshold_met")
        self.assertNotEqual(attempts[0]["prompt_sha256"], attempts[1]["prompt_sha256"])

    def test_equal_or_worse_candidate_does_not_replace_best(self) -> None:
        watermarker, _ = self.watermarker(["original alpha", "original beta"])
        result = watermarker.insert_watermark("original")

        self.assertEqual(result["text2"], "original alpha")
        attempts = result["diagnostics"]["attempts"]
        self.assertTrue(attempts[0]["selected"])
        self.assertFalse(attempts[1]["selected"])
        self.assertEqual(attempts[1]["stop_reason"], "no_improvement")

    def test_all_empty_or_error_generations_fall_back(self) -> None:
        watermarker, inserter = self.watermarker([GenerationError("offline failure")])
        result = watermarker.insert_watermark("original")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["text2"], "original")
        self.assertEqual(result["diagnostics"]["failure_reason"], "generation_failed")
        self.assertEqual(len(inserter.prompts), 1)


if __name__ == "__main__":
    unittest.main()
