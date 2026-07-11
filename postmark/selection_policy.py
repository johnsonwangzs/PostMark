"""Shared watermark-word count policy for insertion and blind detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .common import ConfigurationError, sha256_json


class WordSelector(Protocol):
    selection_config_sha256: str
    selection_config: Mapping[str, Any]

    def word_count_to_k(self, text: str) -> int: ...

    def select_words(self, text: str, *, top_k: int | None = None) -> list[str]: ...


class SelectionPolicy:
    def __init__(
        self,
        selector: WordSelector,
        *,
        min_watermark_words: int | None = None,
        max_watermark_words: int | None = None,
    ) -> None:
        for name, value in (
            ("min_watermark_words", min_watermark_words),
            ("max_watermark_words", max_watermark_words),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ConfigurationError(f"{name} must be a non-negative integer")
        if (
            min_watermark_words is not None
            and max_watermark_words is not None
            and min_watermark_words > max_watermark_words
        ):
            raise ConfigurationError("min_watermark_words exceeds max_watermark_words")
        self.selector = selector
        self.min_watermark_words = min_watermark_words
        self.max_watermark_words = max_watermark_words
        self.config = {
            "selector_selection_config_sha256": selector.selection_config_sha256,
            "selector_selection_config": dict(selector.selection_config),
            "min_watermark_words": min_watermark_words,
            "max_watermark_words": max_watermark_words,
        }
        self.sha256 = sha256_json(self.config)

    def word_count_to_k(self, text: str) -> int:
        k = self.selector.word_count_to_k(text)
        if self.min_watermark_words is not None:
            k = max(k, self.min_watermark_words)
        if self.max_watermark_words is not None:
            k = min(k, self.max_watermark_words)
        return k

    def select_words(self, text: str) -> list[str]:
        return self.selector.select_words(text, top_k=self.word_count_to_k(text))
