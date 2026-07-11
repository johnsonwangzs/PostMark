"""Offline HuggingFace causal language-model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .common import (
    ConfigurationError,
    GenerationError,
    ResourceError,
    load_json_object,
    sha256_json,
)
from .resources import fingerprint_files


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_truncated: bool
    output_truncated: bool
    generated_tokens: int


def _model_snapshot_files(model_path: Path) -> list[str]:
    files = ["config.json"]
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_path / index_name
        if index_path.is_file():
            index = load_json_object(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ResourceError(f"Invalid model weight index: {index_path}")
            files.append(index_name)
            files.extend(sorted(set(weight_map.values())))
            break
    else:
        for weight_name in ("model.safetensors", "pytorch_model.bin"):
            if (model_path / weight_name).is_file():
                files.append(weight_name)
                break
        else:
            raise ResourceError(f"No supported model weights found in {model_path}")
    for optional in ("generation_config.json",):
        if (model_path / optional).is_file():
            files.append(optional)
    files.extend(path.name for path in sorted(model_path.glob("*.py")))
    return sorted(set(files))


def _tokenizer_snapshot_files(tokenizer_path: Path) -> list[str]:
    files = [
        name
        for name in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.json",
        )
        if (tokenizer_path / name).is_file()
    ]
    if not files:
        raise ResourceError(f"No tokenizer files found in {tokenizer_path}")
    return files


class LocalHFLLM:
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        local_files_only: bool = True,
    ) -> None:
        if not local_files_only:
            raise ConfigurationError("PostMark-Local requires local_files_only=True")
        if torch_dtype not in DTYPES:
            raise ConfigurationError(f"Unsupported torch dtype: {torch_dtype}")
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path or model_path)
        if not self.model_path.is_dir() or not self.tokenizer_path.is_dir():
            raise ResourceError("Model and tokenizer must be provisioned local directories")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=DTYPES[torch_dtype],
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise ResourceError(
                "Cannot load local HuggingFace model. Verify the snapshot is complete and "
                "offline dependencies are installed."
            ) from exc
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ResourceError("Tokenizer has neither pad nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if use_chat_template and not getattr(self.tokenizer, "chat_template", None):
            raise ResourceError("use_chat_template=True but tokenizer has no chat template")

        self.model.eval()
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.model_fingerprint = fingerprint_files(
            self.model_path, _model_snapshot_files(self.model_path)
        )
        self.tokenizer_fingerprint = fingerprint_files(
            self.tokenizer_path, _tokenizer_snapshot_files(self.tokenizer_path)
        )
        self.load_config = {
            "model_fingerprint": self.model_fingerprint.to_dict(),
            "tokenizer_fingerprint": self.tokenizer_fingerprint.to_dict(),
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "use_chat_template": use_chat_template,
            "snapshot_generation_defaults_disabled": True,
            "local_files_only": True,
        }
        self.fingerprint_sha256 = sha256_json(self.load_config)

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {**self.load_config, "sha256": self.fingerprint_sha256}

    @property
    def input_device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, RuntimeError) as exc:
            raise ResourceError("Cannot determine local model input device") from exc

    def _render_prompt(self, prompt: str) -> tuple[str, bool]:
        if self.use_chat_template:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered, False
        return prompt, True

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        do_sample: bool | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ConfigurationError("Generation prompt must be a non-empty string")
        if max_new_tokens < 1:
            raise ConfigurationError("max_new_tokens must be positive")
        if temperature < 0:
            raise ConfigurationError("temperature cannot be negative")
        if not 0 < top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1]")
        sample = temperature > 0 if do_sample is None else do_sample
        if sample and temperature <= 0:
            raise ConfigurationError("Sampling requires temperature > 0")

        rendered, add_special_tokens = self._render_prompt(prompt)
        context_window = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(context_window, int) or context_window < 2:
            context_window = 4096
        max_input_tokens = max(1, context_window - max_new_tokens)
        full_ids = self.tokenizer(
            rendered,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
        )["input_ids"]
        input_truncated = len(full_ids) > max_input_tokens
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        ).to(self.input_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})

        cuda_devices: list[int] = []
        if self.input_device.type == "cuda":
            cuda_devices = [
                self.input_device.index
                if self.input_device.index is not None
                else torch.cuda.current_device()
            ]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                if seed is not None:
                    torch.manual_seed(seed)
                with torch.inference_mode():
                    output = self.model.generate(**encoded, **generation_kwargs)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise GenerationError(f"Local generation failed: {exc}") from exc

        input_length = encoded["input_ids"].shape[-1]
        generated = output[0, input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        eos_ids = self.tokenizer.eos_token_id
        if eos_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = set(eos_ids)
        ended_with_eos = bool(generated.numel()) and int(generated[-1]) in eos_set
        output_truncated = generated.numel() >= max_new_tokens and not ended_with_eos
        return GenerationResult(
            text=text,
            input_truncated=input_truncated,
            output_truncated=output_truncated,
            generated_tokens=int(generated.numel()),
        )
