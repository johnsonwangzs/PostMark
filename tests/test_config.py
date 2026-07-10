import copy
import tempfile
import unittest
from pathlib import Path

from postmark.common import ConfigurationError, load_json_object
from postmark.config import PostMarkConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "postmark_portable.json"


class PostMarkConfigTests(unittest.TestCase):
    def test_repository_config_loads(self):
        config = PostMarkConfig.load(CONFIG_PATH)
        self.assertEqual(config.selection_mode, "official_two_stage")
        self.assertEqual(config.detector.presence_mode, "exact_lemma")
        self.assertEqual(config.embedding.max_length, 512)
        self.assertEqual(len(config.sha256), 64)

    def test_unknown_fields_are_rejected(self):
        value = load_json_object(CONFIG_PATH)
        value["unexpected"] = True
        with self.assertRaises(ConfigurationError):
            PostMarkConfig.from_dict(value)

    def test_online_flags_are_rejected(self):
        value = load_json_object(CONFIG_PATH)
        value["runtime"]["offline"] = False
        with self.assertRaises(ConfigurationError):
            PostMarkConfig.from_dict(value)

    def test_invalid_ranges_are_rejected(self):
        original = load_json_object(CONFIG_PATH)
        for section, key, invalid in (
            ("selection", "ratio", 1.1),
            ("anchor", "chunk_words", 0),
            ("insertion", "min_group_presence", -0.1),
        ):
            with self.subTest(section=section, key=key):
                value = copy.deepcopy(original)
                value[section][key] = invalid
                with self.assertRaises(ConfigurationError):
                    PostMarkConfig.from_dict(value)

    def test_relative_paths_resolve_from_explicit_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("inserter", "embedder", "tokenizer"):
                (root / name).mkdir()
            for name in ("corpus.jsonl", "words.pkl", "insert.txt"):
                (root / name).write_text("fixture", encoding="utf-8")
            value = load_json_object(CONFIG_PATH)
            value["paths"] = {
                "inserter": "inserter",
                "embedder": "embedder",
                "embedder_tokenizer": "tokenizer",
                "anchor_corpus": "corpus.jsonl",
                "candidate_words_legacy": "words.pkl",
                "insertion_prompt": "insert.txt",
            }
            resolved = PostMarkConfig.from_dict(value).validate_local_resources(root)
            self.assertEqual(resolved["inserter"], root / "inserter")


if __name__ == "__main__":
    unittest.main()
