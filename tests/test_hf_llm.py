from __future__ import annotations

import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import tempfile
import sys

import torch

from postmark.common import ConfigurationError
from postmark.hf_llm import LocalHFLLM


class _Batch(dict):
    def to(self, device: torch.device) -> "_Batch":
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "fixture"

    def __init__(self) -> None:
        self.rendered: list[str] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.rendered.append(messages[0]["content"])
        self.template_kwargs = kwargs
        return "CHAT:" + messages[0]["content"]

    def __call__(self, text: str, **kwargs):
        ids = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            max_length = kwargs.get("max_length", len(ids))
            ids = ids[-max_length:]
            tensor = torch.tensor([ids], dtype=torch.long)
            return _Batch(
                {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                }
            )
        return {"input_ids": ids}

    def decode(self, ids: torch.Tensor, **kwargs) -> str:
        self.decode_kwargs = kwargs
        return " generated text "


class _Config:
    max_position_embeddings = 8


class _Model:
    config = _Config()

    def __init__(self, generated: list[int]) -> None:
        self.generated = generated
        self.calls: list[dict] = []
        self.embedding = torch.nn.Embedding(10, 4)

    def get_input_embeddings(self):
        return self.embedding

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([self.generated], dtype=torch.long)
        return torch.cat((input_ids, suffix), dim=1)


def _llm(generated: list[int]) -> LocalHFLLM:
    llm = LocalHFLLM.__new__(LocalHFLLM)
    llm.tokenizer = _Tokenizer()
    llm.model = _Model(generated)
    llm.use_chat_template = True
    return llm


class LocalHFGenerationTests(unittest.TestCase):
    def test_loader_is_strictly_local_and_disables_snapshot_sampling_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"fixture")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            tokenizer = _Tokenizer()
            model = _Model([2])
            model.generation_config = SimpleNamespace(
                do_sample=True, temperature=0.6, top_p=0.9, top_k=50
            )
            tokenizer_calls = []
            model_calls = []

            class AutoTokenizer:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    tokenizer_calls.append((path, kwargs))
                    return tokenizer

            class AutoModelForCausalLM:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    model_calls.append((path, kwargs))
                    return model

            module = ModuleType("transformers")
            module.AutoTokenizer = AutoTokenizer
            module.AutoModelForCausalLM = AutoModelForCausalLM
            fingerprint = SimpleNamespace(to_dict=lambda: {"sha256": "fixture"})
            with patch.dict(sys.modules, {"transformers": module}), patch(
                "postmark.hf_llm.fingerprint_files", return_value=fingerprint
            ):
                loaded = LocalHFLLM(str(snapshot), torch_dtype="bfloat16")

        self.assertTrue(tokenizer_calls[0][1]["local_files_only"])
        self.assertTrue(model_calls[0][1]["local_files_only"])
        self.assertIs(model_calls[0][1]["dtype"], torch.bfloat16)
        self.assertNotIn("torch_dtype", model_calls[0][1])
        self.assertFalse(loaded.model.generation_config.do_sample)
        self.assertIsNone(loaded.model.generation_config.temperature)
        self.assertIsNone(loaded.model.generation_config.top_p)
        self.assertIsNone(loaded.model.generation_config.top_k)

    def test_greedy_generation_omits_sampling_parameters(self) -> None:
        llm = _llm([7, 2])
        result = llm.generate(
            "one two three four five six seven",
            max_new_tokens=2,
            temperature=0.0,
        )

        call = llm.model.calls[0]
        self.assertFalse(call["do_sample"])
        self.assertNotIn("temperature", call)
        self.assertNotIn("top_p", call)
        self.assertEqual(result.text, "generated text")
        self.assertTrue(result.input_truncated)
        self.assertFalse(result.output_truncated)
        self.assertEqual(result.generated_tokens, 2)
        self.assertEqual(llm.tokenizer.rendered, ["one two three four five six seven"])

    def test_sampling_passes_parameters_and_marks_length_stop(self) -> None:
        llm = _llm([7, 8])
        result = llm.generate(
            "short prompt",
            max_new_tokens=2,
            temperature=0.7,
            top_p=0.8,
            seed=123,
        )

        call = llm.model.calls[0]
        self.assertTrue(call["do_sample"])
        self.assertEqual(call["temperature"], 0.7)
        self.assertEqual(call["top_p"], 0.8)
        self.assertTrue(result.output_truncated)

    def test_invalid_sampling_configuration_fails_before_generation(self) -> None:
        llm = _llm([2])
        with self.assertRaisesRegex(ConfigurationError, "temperature"):
            llm.generate(
                "prompt",
                max_new_tokens=1,
                temperature=0.0,
                do_sample=True,
            )
        self.assertEqual(llm.model.calls, [])


if __name__ == "__main__":
    unittest.main()
