from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from postmark.common import ConfigurationError
from postmark.presence import ExactLemmaPresence, NomicFuzzyPresence


@dataclass
class _Token:
    text: str
    lemma_: str
    is_space: bool = False
    is_punct: bool = False


class _NLP:
    meta = {"lang": "en", "name": "fixture", "version": "1"}
    spacy_version = "fixture"

    lemmas = {
        "ran": "run",
        "running": "run",
        "runs": "run",
        "children": "child",
    }

    def __call__(self, text: str):
        tokens = []
        for raw in text.replace(",", " , ").replace("!", " ! ").split():
            lower = raw.lower()
            tokens.append(
                _Token(
                    raw,
                    self.lemmas.get(lower, lower),
                    is_punct=raw in {",", "!"},
                )
            )
        return tokens


def _presence() -> ExactLemmaPresence:
    return ExactLemmaPresence(
        "fixture",
        _nlp=_NLP(),
        _resource_fingerprint={"sha256": "fixture"},
    )


class ExactLemmaPresenceTests(unittest.TestCase):
    def test_matches_surface_case_and_lemma(self) -> None:
        result = _presence().score(
            "The CHILDREN ran, and a Scale remained!",
            ["child", "running", "scale", "missing"],
        )
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.present_words, ["child", "running", "scale"])
        self.assertEqual(result.missing_words, ["missing"])
        self.assertNotEqual(result.token_form_count, 0)

    def test_empty_expected_list_has_fixed_zero_score(self) -> None:
        result = _presence().score("any text", [])
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.present_words, [])
        self.assertEqual(result.missing_words, [])

    def test_duplicate_or_empty_expected_words_are_rejected(self) -> None:
        presence = _presence()
        for words in (["run", "RUN"], [""]):
            with self.subTest(words=words), self.assertRaises(ConfigurationError):
                presence.score("text", words)

    def test_fingerprint_binds_normalization_and_spacy_resource(self) -> None:
        presence = _presence()
        self.assertEqual(presence.fingerprint["presence_mode"], "exact_lemma")
        self.assertEqual(
            presence.fingerprint["spacy_model"]["fingerprint"]["sha256"],
            "fixture",
        )
        self.assertEqual(len(presence.fingerprint["sha256"]), 64)


class _Encoder:
    vectors = {
        "fast": [1.0, 0.0],
        "rock": [0.0, 1.0],
        "absent": [-1.0, 0.0],
        "swiftly": [0.8, 0.6],
        "stone": [0.0, 1.0],
    }

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        self.calls.append(texts.copy())
        return torch.tensor([self.vectors.get(text, [0.2, 0.2]) for text in texts])


class NomicFuzzyPresenceTests(unittest.TestCase):
    def test_fuzzy_matching_runs_after_exact_and_caches_expected_words(self) -> None:
        encoder = _Encoder()
        presence = NomicFuzzyPresence(
            _presence(),
            encoder,
            encoder_fingerprint={"sha256": "nomic-fixture"},
            similarity_threshold=0.75,
        )
        first = presence.score("swiftly stone", ["fast", "rock", "absent"])
        second = presence.score("swiftly stone", ["fast", "rock", "absent"])

        self.assertEqual(first.score, 2 / 3)
        self.assertEqual(first.present_words, ["fast", "rock"])
        self.assertEqual(first.missing_words, ["absent"])
        self.assertEqual(second, first)
        self.assertEqual(
            sum(call == ["fast", "rock", "absent"] for call in encoder.calls),
            1,
        )
        self.assertEqual(presence.fingerprint["presence_mode"], "nomic_fuzzy")


if __name__ == "__main__":
    unittest.main()
