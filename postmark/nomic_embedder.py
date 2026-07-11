"""Local Nomic text encoding with the fixed PostMark pooling contract."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .common import ConfigurationError, ResourceError
from .resources import PathFingerprint, fingerprint_files


class NomicTextEncoder:
    def __init__(
        self,
        embedder_path: str,
        *,
        tokenizer_path: str,
        max_length: int = 512,
        task_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        if not local_files_only:
            raise ConfigurationError("PostMark-Local requires local_files_only=True")
        if max_length < 1 or batch_size < 1:
            raise ConfigurationError("max_length and batch_size must be positive")
        self.embedder_path = Path(embedder_path)
        self.tokenizer_path = Path(tokenizer_path)
        if not self.embedder_path.is_dir() or not self.tokenizer_path.is_dir():
            raise ResourceError("Nomic model and tokenizer paths must be local directories")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.embedder_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.task_prefix = task_prefix
        self.batch_size = batch_size

    @property
    def embedding_dim(self) -> int:
        dimension = getattr(self.model.config, "hidden_size", None)
        if not isinstance(dimension, int):
            raise ResourceError("Cannot determine Nomic embedding dimension")
        return dimension

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if not isinstance(texts, list) or not texts:
            raise ConfigurationError("encode_texts requires a non-empty list")
        if any(not isinstance(text, str) for text in texts):
            raise ConfigurationError("All encoder inputs must be strings")

        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = [
                self.task_prefix + text for text in texts[start : start + self.batch_size]
            ]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                model_output = self.model(**encoded)
            token_embeddings = model_output[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            mask = mask.to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normalized = F.normalize(pooled, p=2, dim=1)
            outputs.append(normalized.detach().cpu().to(torch.float32))
        return torch.cat(outputs, dim=0)

    def model_fingerprint(self) -> PathFingerprint:
        files = ["config.json"]
        if (self.embedder_path / "model.safetensors").is_file():
            files.append("model.safetensors")
        elif (self.embedder_path / "pytorch_model.bin").is_file():
            files.append("pytorch_model.bin")
        else:
            raise ResourceError("Nomic snapshot has no supported model weights")
        for optional in (
            "configuration_hf_nomic_bert.py",
            "modeling_hf_nomic_bert.py",
        ):
            if (self.embedder_path / optional).is_file():
                files.append(optional)
        return fingerprint_files(self.embedder_path, files)

    def tokenizer_fingerprint(self) -> PathFingerprint:
        files = [
            name
            for name in (
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt",
                "special_tokens_map.json",
            )
            if (self.tokenizer_path / name).is_file()
        ]
        if not files:
            raise ResourceError("Tokenizer snapshot contains no recognized tokenizer files")
        return fingerprint_files(self.tokenizer_path, files)
