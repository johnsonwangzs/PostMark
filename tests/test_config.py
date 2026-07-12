import copy
import tempfile
import unittest
from pathlib import Path

from postmark.common import ConfigurationError, load_json_object
from postmark.config import PostMarkConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "postmark_portable.json"
EXPERIMENT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "postmark_200.json"
EXPERIMENT_PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "postmark_200_protocol.json"


class PostMarkConfigTests(unittest.TestCase):
    def test_repository_config_loads(self):
        config = PostMarkConfig.load(CONFIG_PATH)
        self.assertEqual(config.selection_mode, "official_two_stage")
        self.assertEqual(config.detector.presence_mode, "exact_lemma")
        self.assertEqual(config.embedding.max_length, 512)
        self.assertEqual(len(config.sha256), 64)

    def test_frozen_200_pair_config_loads(self):
        config = PostMarkConfig.load(EXPERIMENT_CONFIG_PATH)
        self.assertEqual(config.selection.ratio, 0.06)
        self.assertEqual(config.insertion.group_size, 20)
        self.assertEqual(config.insertion.max_new_tokens, 768)
        self.assertEqual(config.detector.presence_mode, "nomic_fuzzy")
        self.assertEqual(config.detector.similarity_threshold, 0.8)
        self.assertEqual(config.detector.max_content_tokens, 128)
        self.assertEqual(config.runtime.seed, 1618)

    def test_frozen_protocol_binds_current_config(self):
        config = PostMarkConfig.load(EXPERIMENT_CONFIG_PATH)
        protocol = load_json_object(EXPERIMENT_PROTOCOL_PATH)
        self.assertEqual(protocol["postmark_config_sha256"], config.sha256)
        self.assertEqual(protocol["formal_test_status"], "not_run")
        self.assertEqual(protocol["calibration"]["negative_count"], 1000)
        self.assertEqual(protocol["calibration"]["target_fpr"], 0.01)
        self.assertFalse(protocol["paragram_in_scope"])

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
            for name in ("candidate_words.json", "nomic_table.pt"):
                (root / name).write_text("fixture", encoding="utf-8")
            value = load_json_object(CONFIG_PATH)
            value["paths"] = {
                "inserter": "inserter",
                "embedder": "embedder",
                "embedder_tokenizer": "tokenizer",
                "anchor_corpus": "corpus.jsonl",
                "candidate_words_legacy": "words.pkl",
                "candidate_words": "candidate_words.json",
                "nomic_table": "nomic_table.pt",
                "insertion_prompt": "insert.txt",
            }
            resolved = PostMarkConfig.from_dict(value).validate_local_resources(root)
            self.assertEqual(resolved["inserter"], root / "inserter")


if __name__ == "__main__":
    unittest.main()
