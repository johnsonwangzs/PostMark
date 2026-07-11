"""Local token-presence implementations for blind PostMark detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ConfigurationError, ResourceError, sha256_json
from .resources import fingerprint_path


EXACT_LEMMA_VERSION = 1


@dataclass(frozen=True)
class PresenceResult:
    score: float
    present_words: list[str]
    missing_words: list[str]
    token_form_count: int


def _token_forms(doc: Any) -> set[str]:
    forms: set[str] = set()
    for token in doc:
        if bool(getattr(token, "is_space", False)) or bool(
            getattr(token, "is_punct", False)
        ):
            continue
        text = str(getattr(token, "text", "")).strip().lower()
        lemma = str(getattr(token, "lemma_", "")).strip().lower()
        if text:
            forms.add(text)
        if lemma and lemma != "-pron-":
            forms.add(lemma)
    return forms


class ExactLemmaPresence:
    """Case-insensitive token/lemma presence using a fixed local spaCy pipeline."""

    def __init__(
        self,
        spacy_model: str,
        *,
        _nlp: Any | None = None,
        _resource_fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(spacy_model, str) or not spacy_model:
            raise ConfigurationError("spacy_model must be a non-empty local package or path")
        if (_nlp is None) != (_resource_fingerprint is None):
            raise ConfigurationError(
                "Injected spaCy pipelines require an explicit resource fingerprint"
            )
        if _nlp is None:
            try:
                import spacy

                self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
            except (ImportError, OSError, ValueError) as exc:
                raise ResourceError(
                    f"Cannot load local spaCy model {spacy_model!r}. Install its wheel "
                    "offline or provide a local model directory."
                ) from exc
            model_path = Path(self.nlp.path)
            if not model_path.is_dir():
                raise ResourceError(
                    f"spaCy model {spacy_model!r} did not resolve to a local directory"
                )
            resource_fingerprint = fingerprint_path(model_path).to_dict()
            spacy_version = spacy.__version__
        else:
            self.nlp = _nlp
            resource_fingerprint = dict(_resource_fingerprint)
            spacy_version = str(getattr(_nlp, "spacy_version", "fixture"))

        meta = getattr(self.nlp, "meta", {})
        self.config = {
            "version": EXACT_LEMMA_VERSION,
            "presence_mode": "exact_lemma",
            "spacy_model": {
                "lang": meta.get("lang"),
                "name": meta.get("name"),
                "version": meta.get("version"),
                "spacy_version": spacy_version,
                "fingerprint": resource_fingerprint,
            },
            "normalization": {
                "lowercase": True,
                "include_surface": True,
                "include_lemma": True,
                "drop_space": True,
                "drop_punctuation": True,
                "drop_stop_words": False,
            },
        }
        self.fingerprint_sha256 = sha256_json(self.config)
        self._expected_form_cache: dict[str, set[str]] = {}

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {**self.config, "sha256": self.fingerprint_sha256}

    def _expected_forms(self, word: str) -> set[str]:
        normalized = word.strip().lower()
        if not normalized:
            raise ConfigurationError("Expected watermark words must be non-empty strings")
        cached = self._expected_form_cache.get(normalized)
        if cached is None:
            cached = _token_forms(self.nlp(normalized))
            cached.add(normalized)
            self._expected_form_cache[normalized] = cached
        return cached

    def score(self, text: str, expected_words: Sequence[str]) -> PresenceResult:
        if not isinstance(text, str):
            raise ConfigurationError("Presence input text must be a string")
        if not isinstance(expected_words, Sequence) or isinstance(expected_words, str):
            raise ConfigurationError("expected_words must be a sequence of strings")
        words = list(expected_words)
        if any(not isinstance(word, str) or not word.strip() for word in words):
            raise ConfigurationError("Expected watermark words must be non-empty strings")
        normalized_words = [word.strip().lower() for word in words]
        if len(set(normalized_words)) != len(normalized_words):
            raise ConfigurationError("Expected watermark words must be unique")
        if not words:
            return PresenceResult(0.0, [], [], 0)

        text_forms = _token_forms(self.nlp(text))
        present: list[str] = []
        missing: list[str] = []
        for word, normalized in zip(words, normalized_words):
            destination = (
                present
                if self._expected_forms(normalized) & text_forms
                else missing
            )
            destination.append(word)
        return PresenceResult(
            score=len(present) / len(words),
            present_words=present,
            missing_words=missing,
            token_form_count=len(text_forms),
        )


class NomicFuzzyPresence:
    """Exact lemma matching followed by local Nomic cosine matching."""

    def __init__(
        self,
        exact_presence: ExactLemmaPresence,
        encoder: Any,
        *,
        encoder_fingerprint: Mapping[str, Any],
        similarity_threshold: float = 0.75,
        max_content_tokens: int = 128,
        min_token_length: int = 3,
    ) -> None:
        if (
            isinstance(similarity_threshold, bool)
            or not isinstance(similarity_threshold, (int, float))
            or not -1 <= similarity_threshold <= 1
        ):
            raise ConfigurationError("similarity_threshold must be in [-1, 1]")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (max_content_tokens, min_token_length)
        ):
            raise ConfigurationError("Nomic token limits must be positive integers")
        if not callable(getattr(encoder, "encode_texts", None)):
            raise ConfigurationError("Nomic presence encoder must provide encode_texts")
        self.exact_presence = exact_presence
        self.encoder = encoder
        self.similarity_threshold = float(similarity_threshold)
        self.max_content_tokens = max_content_tokens
        self.min_token_length = min_token_length
        self.config = {
            "version": 1,
            "presence_mode": "nomic_fuzzy",
            "exact_lemma_fingerprint": exact_presence.fingerprint,
            "nomic_encoder_fingerprint": dict(encoder_fingerprint),
            "similarity_threshold": self.similarity_threshold,
            "token_filter": {
                "lowercase": True,
                "prefer_lemma": True,
                "drop_stop_words": True,
                "drop_space": True,
                "drop_punctuation": True,
                "alphabetic_only": True,
                "min_token_length": min_token_length,
                "max_content_tokens": max_content_tokens,
                "deduplicate_preserve_order": True,
            },
            "oov_rule": "nomic_has_no_explicit_oov; exact_lemma_runs_first",
        }
        self.fingerprint_sha256 = sha256_json(self.config)
        self._expected_embedding_cache: dict[str, Any] = {}

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {**self.config, "sha256": self.fingerprint_sha256}

    def _content_tokens(self, text: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in self.exact_presence.nlp(text):
            if (
                bool(getattr(token, "is_space", False))
                or bool(getattr(token, "is_punct", False))
                or bool(getattr(token, "is_stop", False))
            ):
                continue
            surface = str(getattr(token, "text", "")).strip().lower()
            lemma = str(getattr(token, "lemma_", "")).strip().lower()
            normalized = lemma if lemma and lemma != "-pron-" else surface
            is_alpha = bool(getattr(token, "is_alpha", normalized.isalpha()))
            if (
                not is_alpha
                or len(normalized) < self.min_token_length
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            result.append(normalized)
            if len(result) >= self.max_content_tokens:
                break
        return result

    def _expected_embeddings(self, words: list[str]) -> Any:
        import torch

        missing_cache = [word for word in words if word not in self._expected_embedding_cache]
        if missing_cache:
            encoded = self.encoder.encode_texts(missing_cache)
            if encoded.ndim != 2 or encoded.shape[0] != len(missing_cache):
                raise ResourceError("Nomic presence encoder returned an invalid tensor shape")
            if not torch.isfinite(encoded).all():
                raise ResourceError("Nomic presence encoder returned NaN or Inf")
            normalized = torch.nn.functional.normalize(
                encoded.detach().cpu().to(torch.float32), p=2, dim=1
            )
            for word, vector in zip(missing_cache, normalized):
                self._expected_embedding_cache[word] = vector
        return torch.stack([self._expected_embedding_cache[word] for word in words])

    def score(self, text: str, expected_words: Sequence[str]) -> PresenceResult:
        import torch

        exact = self.exact_presence.score(text, expected_words)
        if not exact.missing_words:
            return exact
        content_tokens = self._content_tokens(text)
        if not content_tokens:
            return exact
        expected = self._expected_embeddings(
            [word.strip().lower() for word in exact.missing_words]
        )
        content = self.encoder.encode_texts(content_tokens)
        if content.ndim != 2 or content.shape[0] != len(content_tokens):
            raise ResourceError("Nomic presence encoder returned an invalid tensor shape")
        if expected.shape[1] != content.shape[1] or not torch.isfinite(content).all():
            raise ResourceError("Nomic presence embedding dimensions or values are invalid")
        content = torch.nn.functional.normalize(
            content.detach().cpu().to(torch.float32), p=2, dim=1
        )
        similarities = expected @ content.T
        fuzzy_words = {
            word
            for word, maximum in zip(exact.missing_words, similarities.max(dim=1).values)
            if float(maximum) >= self.similarity_threshold
        }
        present_set = set(exact.present_words) | fuzzy_words
        words = list(expected_words)
        present = [word for word in words if word in present_set]
        missing = [word for word in words if word not in present_set]
        return PresenceResult(
            score=len(present) / len(words) if words else 0.0,
            present_words=present,
            missing_words=missing,
            token_form_count=exact.token_form_count,
        )
