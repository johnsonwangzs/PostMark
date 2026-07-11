from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from postmark.common import load_jsonl
from postmark.watermark import PostMarkWatermarker, run_watermark_pipeline
from tests.watermark_fixtures import EchoInserter, FakeSelector, SamplingBase


class JsonlPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prompt = self.root / "insert.txt"
        self.prompt.write_text("TEXT={}\nWORDS={}", encoding="utf-8")
        self.output = self.root / "output.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def watermarker(self) -> PostMarkWatermarker:
        return PostMarkWatermarker(
            EchoInserter(),
            FakeSelector(),
            prompt_path=str(self.prompt),
            group_size=2,
            min_group_presence=1.0,
            max_new_tokens=20,
        )

    def write_input(self, records) -> Path:
        path = self.root / "input.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    def test_text_pipeline_writes_fields_manifest_and_durable_newline(self) -> None:
        input_path = self.write_input([{"id": "a", "text": "original text"}])
        result = run_watermark_pipeline(
            input_path=str(input_path),
            output_path=str(self.output),
            watermarker=self.watermarker(),
            text_field="text",
        )

        self.assertEqual(result, {"input": 1, "written": 1, "skipped": 0})
        self.assertTrue(self.output.read_bytes().endswith(b"\n"))
        record = load_jsonl(self.output)[0]
        for field in (
            "id", "status", "text1", "list1", "text2", "list2", "diagnostics",
            "input_sha256", "selection_config_sha256", "run_config_sha256",
            "selector_resource_sha256",
        ):
            self.assertIn(field, record)
        manifest = json.loads((self.root / "output.jsonl.manifest.json").read_text())
        self.assertEqual(manifest["run_config_sha256"], record["run_config_sha256"])

    def test_prompt_mode_uses_stable_per_id_seed(self) -> None:
        input_path = self.write_input(
            [{"id": "b", "prompt": "second"}, {"id": "a", "prompt": "first"}]
        )
        base = SamplingBase()
        run_watermark_pipeline(
            input_path=str(input_path),
            output_path=str(self.output),
            watermarker=self.watermarker(),
            prompt_field="prompt",
            base_llm=base,
            seed=9,
        )
        seeds = {prompt: seed for prompt, seed in base.calls}
        self.assertEqual(len(set(seeds.values())), 2)

        reordered = self.write_input(
            [{"id": "a", "prompt": "first"}, {"id": "b", "prompt": "second"}]
        )
        other_output = self.root / "other.jsonl"
        other_base = SamplingBase()
        run_watermark_pipeline(
            input_path=str(reordered),
            output_path=str(other_output),
            watermarker=self.watermarker(),
            prompt_field="prompt",
            base_llm=other_base,
            seed=9,
        )
        self.assertEqual(seeds, {prompt: seed for prompt, seed in other_base.calls})


if __name__ == "__main__":
    unittest.main()
