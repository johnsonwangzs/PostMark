import json
import pickle
import tempfile
import unittest
from pathlib import Path

from postmark.build_candidate_words import build_compat_candidate_words
from postmark.common import ConfigurationError, ResourceMismatchError
from postmark.resources import load_candidate_words, write_candidate_words


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_WORDS = REPOSITORY_ROOT / "valid_wtmk_words_in_wiki_base-only-f1000.pkl"


class _UnsafePicklePayload:
    def __reduce__(self):
        return (str, ("unsafe",))


class CandidateWordsTests(unittest.TestCase):
    def test_repository_compat_words_match_expected_contract(self):
        resource = build_compat_candidate_words(LEGACY_WORDS)
        self.assertEqual(len(resource.words), 3266)
        self.assertEqual(resource.words[:5], ["not", "first", "also", "have", "time"])
        self.assertEqual(
            resource.source_sha256,
            "37455b6f37e8580e61aa19aaa448beccdcffa97b20cc1f25e2c70fddb617fbe7",
        )

    def test_round_trip_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "words.pkl"
            output = root / "words.json"
            words = ["zeta", "alpha", "middle"]
            source.write_bytes(pickle.dumps(words))
            resource = build_compat_candidate_words(source, expected_word_count=3)
            write_candidate_words(output, resource)
            self.assertEqual(load_candidate_words(output).words, words)

    def test_duplicate_words_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "words.pkl"
            source.write_bytes(pickle.dumps(["same", "same"]))
            with self.assertRaises(ConfigurationError):
                build_compat_candidate_words(source, expected_word_count=2)

    def test_pickle_globals_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "words.pkl"
            source.write_bytes(pickle.dumps([_UnsafePicklePayload()]))
            with self.assertRaises(ConfigurationError):
                build_compat_candidate_words(source, expected_word_count=1)

    def test_tampered_words_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "words.pkl"
            output = root / "words.json"
            source.write_bytes(pickle.dumps(["alpha", "beta"]))
            resource = build_compat_candidate_words(source, expected_word_count=2)
            write_candidate_words(output, resource)
            value = json.loads(output.read_text(encoding="utf-8"))
            value["words"][0] = "changed"
            output.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ResourceMismatchError):
                load_candidate_words(output)


if __name__ == "__main__":
    unittest.main()
