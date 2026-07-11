"""Local Nomic text encoding with the fixed PostMark pooling contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .common import (
    ConfigurationError,
    ResourceError,
    ResourceMismatchError,
    SelectionError,
    sha256_json,
    stable_word_count,
)
from .resources import PathFingerprint, fingerprint_files


SELECTION_CONFIG_VERSION = 1
SELECTION_MODES = {"official_two_stage", "anchor_only", "direct_word"}


@dataclass(frozen=True)
class SelectionResult:
    words: list[str]
    indices: list[int]
    prefilter_indices: list[int]
    k: int
    selection_mode: str


def stable_topk(
    scores: torch.Tensor,
    k: int,
    *,
    tie_break_indices: list[int] | None = None,
) -> list[int]:
    """Return score positions ordered by score desc, then stable candidate index."""

    if scores.ndim != 1:
        raise SelectionError("stable_topk scores must be one-dimensional")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > scores.numel():
        raise SelectionError(f"Invalid top-k value {k} for {scores.numel()} scores")
    if not torch.isfinite(scores).all():
        raise SelectionError("Selection scores contain NaN or Inf")
    if tie_break_indices is None:
        tie_break_indices = list(range(scores.numel()))
    if len(tie_break_indices) != scores.numel() or len(set(tie_break_indices)) != len(
        tie_break_indices
    ):
        raise SelectionError("tie_break_indices must be unique and aligned with scores")

    cpu_scores = scores.detach().cpu().to(torch.float64).tolist()
    positions = range(len(cpu_scores))
    return sorted(
        positions,
        key=lambda position: (-cpu_scores[position], tie_break_indices[position]),
    )[:k]


def select_candidate_indices(
    text_embedding: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    candidate_word_embeddings: torch.Tensor,
    *,
    k: int,
    selection_mode: str = "official_two_stage",
    prefilter_multiplier: int = 3,
) -> tuple[list[int], list[int]]:
    """Select candidate indices from already normalized embedding tensors."""

    if selection_mode not in SELECTION_MODES:
        raise SelectionError(f"Unsupported selection mode: {selection_mode}")
    if prefilter_multiplier < 1:
        raise SelectionError("prefilter_multiplier must be positive")
    if text_embedding.ndim != 1:
        raise SelectionError("text_embedding must be one-dimensional")
    if anchor_embeddings.ndim != 2 or candidate_word_embeddings.ndim != 2:
        raise SelectionError("Selection tables must be two-dimensional")
    vocabulary_size = anchor_embeddings.shape[0]
    if candidate_word_embeddings.shape[0] != vocabulary_size:
        raise SelectionError("Anchor and candidate tables have different row counts")
    if (
        anchor_embeddings.shape[1] != text_embedding.shape[0]
        or candidate_word_embeddings.shape[1] != text_embedding.shape[0]
    ):
        raise SelectionError("Text and table embedding dimensions differ")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise SelectionError("k must be a non-negative integer")
    if k == 0:
        return [], []
    if k > vocabulary_size:
        raise SelectionError(
            f"Requested k={k} exceeds candidate vocabulary size {vocabulary_size}"
        )

    if selection_mode == "direct_word":
        scores = candidate_word_embeddings @ text_embedding
        return stable_topk(scores, k), []

    anchor_scores = anchor_embeddings @ text_embedding
    if selection_mode == "anchor_only":
        indices = stable_topk(anchor_scores, k)
        return indices, indices

    prefilter_k = min(prefilter_multiplier * k, vocabulary_size)
    prefilter_indices = stable_topk(anchor_scores, prefilter_k)
    index_tensor = torch.tensor(
        prefilter_indices,
        dtype=torch.long,
        device=candidate_word_embeddings.device,
    )
    word_scores = candidate_word_embeddings.index_select(0, index_tensor) @ text_embedding
    reranked_positions = stable_topk(
        word_scores,
        k,
        tie_break_indices=prefilter_indices,
    )
    final_indices = [prefilter_indices[position] for position in reranked_positions]
    return final_indices, prefilter_indices


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


class NomicPostMarkEmbedder:
    def __init__(
        self,
        embedder_path: str,
        table_path: str,
        *,
        tokenizer_path: str | None = None,
        implementation_profile: str = "compat",
        selection_mode: str = "official_two_stage",
        ratio: float = 0.12,
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
        local_files_only: bool = True,
        allow_resource_mismatch: bool = False,
        allow_config_mismatch: bool = False,
    ) -> None:
        if implementation_profile != "compat":
            raise ConfigurationError("This selector currently requires compat profile")
        if selection_mode not in SELECTION_MODES:
            raise ConfigurationError(f"Unsupported selection mode: {selection_mode}")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
            raise ConfigurationError("ratio must be in [0, 1]")
        if not local_files_only:
            raise ConfigurationError("PostMark-Local requires local_files_only=True")

        resolved_table_path = Path(table_path)
        if not resolved_table_path.is_file():
            raise ResourceError(
                f"Selector table does not exist: {resolved_table_path}. Build it with "
                "`python -m postmark.build_nomic_anchor_pool`."
            )

        from .build_nomic_anchor_pool import load_table

        table, manifest = load_table(resolved_table_path)
        table_embedder = table["embedder"]
        table_max_length = table_embedder.get("max_length")
        config_mismatches: list[str] = []
        if table.get("implementation_profile") != implementation_profile:
            config_mismatches.append("implementation_profile")
        if table_max_length != max_length:
            config_mismatches.append("max_length")
        if table_embedder.get("pooling") != "mean":
            config_mismatches.append("pooling")
        if table_embedder.get("normalization") != "l2":
            config_mismatches.append("normalization")
        if config_mismatches and not allow_config_mismatch:
            raise ResourceMismatchError(
                "Selector configuration mismatch: " + ", ".join(config_mismatches)
            )

        resolved_tokenizer = tokenizer_path if tokenizer_path is not None else embedder_path
        encoder = NomicTextEncoder(
            embedder_path,
            tokenizer_path=resolved_tokenizer,
            max_length=max_length,
            task_prefix=table_embedder.get("task_prefix", ""),
            batch_size=batch_size,
            device=device,
            local_files_only=True,
        )
        actual_model_fingerprint = encoder.model_fingerprint().to_dict()
        actual_tokenizer_fingerprint = encoder.tokenizer_fingerprint().to_dict()
        resource_mismatches: list[str] = []
        if actual_model_fingerprint != table_embedder.get("fingerprint"):
            resource_mismatches.append("embedder_fingerprint")
        if actual_tokenizer_fingerprint != table_embedder.get("tokenizer_fingerprint"):
            resource_mismatches.append("tokenizer_fingerprint")
        if encoder.embedding_dim != table_embedder.get("embedding_dim"):
            raise ResourceMismatchError(
                "Selector embedding dimension differs from the table and cannot be overridden"
            )
        if resource_mismatches and not allow_resource_mismatch:
            raise ResourceMismatchError(
                "Selector resource mismatch: " + ", ".join(resource_mismatches)
            )

        self.encoder = encoder
        self.table_path = resolved_table_path
        self.table_manifest = manifest
        self.candidate_words = table["candidate_words"]
        self.anchor_embeddings = table["anchor_embeddings"].to(encoder.device)
        self.candidate_word_embeddings = table["candidate_word_embeddings"].to(
            encoder.device
        )
        self.implementation_profile = implementation_profile
        self.selection_mode = selection_mode
        self.ratio = float(ratio)
        self.prefilter_multiplier = table["prefilter_multiplier"]
        self.config_consistent = not config_mismatches and not resource_mismatches
        self.eligible_for_aggregate = self.config_consistent
        self.paper_method_compatible = selection_mode == "official_two_stage"
        self.exact_paper_reproduction = False
        self.selection_config = self._build_selection_config(
            actual_model_fingerprint,
            actual_tokenizer_fingerprint,
            table,
        )
        self.selection_config_sha256 = sha256_json(self.selection_config)

    def _build_selection_config(
        self,
        model_fingerprint: dict[str, Any],
        tokenizer_fingerprint: dict[str, Any],
        table: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "version": SELECTION_CONFIG_VERSION,
            "selector_resource_sha256": self.table_manifest.content_sha256,
            "embedder_fingerprint": model_fingerprint,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "pooling": table["embedder"]["pooling"],
            "normalization": table["embedder"]["normalization"],
            "max_length": self.encoder.max_length,
            "task_prefix": self.encoder.task_prefix,
            "implementation_profile": self.implementation_profile,
            "selection_mode": self.selection_mode,
            "ratio": self.ratio,
            "word_count_rule": "whitespace_split_floor",
            "prefilter_multiplier": self.prefilter_multiplier,
            "mapping_algorithm_version": table["mapping_algorithm_version"],
        }

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        return self.encoder.encode_texts(texts)

    def word_count_to_k(self, text: str, *, ratio: float | None = None) -> int:
        selected_ratio = self.ratio if ratio is None else ratio
        if (
            isinstance(selected_ratio, bool)
            or not isinstance(selected_ratio, (int, float))
            or not 0 <= selected_ratio <= 1
        ):
            raise ConfigurationError("ratio must be in [0, 1]")
        return int(stable_word_count(text) * selected_ratio)

    def select(
        self,
        text: str,
        *,
        ratio: float | None = None,
        top_k: int | None = None,
    ) -> SelectionResult:
        if not isinstance(text, str):
            raise ConfigurationError("text must be a string")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
        ):
            raise ConfigurationError("top_k must be a non-negative integer")
        k = self.word_count_to_k(text, ratio=ratio) if top_k is None else top_k
        if k == 0:
            return SelectionResult([], [], [], 0, self.selection_mode)
        if k > len(self.candidate_words):
            raise SelectionError(
                f"Requested k={k} exceeds candidate vocabulary size "
                f"{len(self.candidate_words)}"
            )

        text_embedding = self.encode_texts([text])[0].to(self.encoder.device)
        indices, prefilter_indices = select_candidate_indices(
            text_embedding,
            self.anchor_embeddings,
            self.candidate_word_embeddings,
            k=k,
            selection_mode=self.selection_mode,
            prefilter_multiplier=self.prefilter_multiplier,
        )
        words = sorted({self.candidate_words[index].lower() for index in indices})
        if len(words) != k:
            raise SelectionError("Selected candidate words are not unique after normalization")
        return SelectionResult(words, indices, prefilter_indices, k, self.selection_mode)

    def select_words(
        self,
        text: str,
        *,
        ratio: float | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        return self.select(text, ratio=ratio, top_k=top_k).words
