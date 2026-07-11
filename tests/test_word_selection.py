import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from postmark.common import ResourceMismatchError, SelectionError
from postmark.nomic_embedder import (
    NomicPostMarkEmbedder,
    SelectionResult,
    select_candidate_indices,
    stable_topk,
)
from postmark.resources import PathFingerprint, ResourceManifest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "official_selector_fixture.json"


def unit_rows(first_components):
    return torch.tensor(
        [
            [component, math.sqrt(max(0.0, 1.0 - component * component))]
            for component in first_components
        ],
        dtype=torch.float32,
    )


class SelectorFixtureTests(unittest.TestCase):
    def test_official_two_stage_fixture(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        text_embedding = torch.tensor(fixture["text_embedding"], dtype=torch.float32)
        anchors = unit_rows(fixture["anchor_first_components"])
        candidates = unit_rows(fixture["candidate_first_components"])
        indices, prefilter = select_candidate_indices(
            text_embedding,
            anchors,
            candidates,
            k=fixture["k"],
            selection_mode="official_two_stage",
            prefilter_multiplier=fixture["prefilter_multiplier"],
        )
        self.assertEqual(prefilter, fixture["expected_prefilter_indices"])
        self.assertEqual(indices, fixture["expected_final_indices"])
        self.assertEqual(
            sorted(fixture["candidate_words"][index] for index in indices),
            fixture["expected_words"],
        )

        anchor_only, _ = select_candidate_indices(
            text_embedding,
            anchors,
            candidates,
            k=fixture["k"],
            selection_mode="anchor_only",
        )
        self.assertEqual(anchor_only, [0, 1])
        self.assertNotEqual(anchor_only, indices)

    def test_stable_topk_uses_candidate_index_for_ties(self):
        positions = stable_topk(
            torch.tensor([0.5, 0.5, 0.5]),
            2,
            tie_break_indices=[5, 2, 9],
        )
        self.assertEqual(positions, [1, 0])

    def test_k_boundaries_are_explicit(self):
        text = torch.tensor([1.0, 0.0])
        table = torch.eye(2)
        self.assertEqual(
            select_candidate_indices(text, table, table, k=0),
            ([], []),
        )
        with self.assertRaises(SelectionError):
            select_candidate_indices(text, table, table, k=3)
        with self.assertRaises(SelectionError):
            stable_topk(torch.tensor([float("nan")]), 1)


class FakeEncoder:
    model_digest = "a" * 64
    tokenizer_digest = "b" * 64

    def __init__(
        self,
        embedder_path,
        *,
        tokenizer_path,
        max_length,
        task_prefix,
        batch_size,
        device,
        local_files_only,
    ):
        self.embedder_path = embedder_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.task_prefix = task_prefix
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.embedding_dim = 2
        self.encode_calls = 0

    def model_fingerprint(self):
        return PathFingerprint(self.model_digest, "file_set", 1, 1)

    def tokenizer_fingerprint(self):
        return PathFingerprint(self.tokenizer_digest, "file_set", 1, 1)

    def encode_texts(self, texts):
        self.encode_calls += 1
        return torch.tensor([[1.0, 0.0] for _ in texts], dtype=torch.float32)


def fake_table(model_digest="a" * 64):
    model_fingerprint = PathFingerprint(model_digest, "file_set", 1, 1).to_dict()
    tokenizer_fingerprint = PathFingerprint("b" * 64, "file_set", 1, 1).to_dict()
    table = {
        "implementation_profile": "compat",
        "candidate_words": ["alpha", "beta"],
        "anchor_embeddings": torch.eye(2),
        "candidate_word_embeddings": torch.eye(2),
        "prefilter_multiplier": 3,
        "mapping_algorithm_version": 1,
        "embedder": {
            "fingerprint": model_fingerprint,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "embedding_dim": 2,
            "pooling": "mean",
            "normalization": "l2",
            "max_length": 512,
            "task_prefix": "",
        },
    }
    manifest = ResourceManifest(
        resource_type="postmark_nomic_table",
        resource_version=2,
        content_sha256="c" * 64,
    )
    return table, manifest


class SelectorResourceContractTests(unittest.TestCase):
    def make_selector(self, load_table, **kwargs):
        load_table.return_value = fake_table()
        with patch("pathlib.Path.is_file", return_value=True):
            return NomicPostMarkEmbedder(
                "/model",
                "/table.pt",
                tokenizer_path="/tokenizer",
                device="cpu",
                **kwargs,
            )

    @patch("postmark.nomic_embedder.NomicTextEncoder", FakeEncoder)
    @patch("postmark.build_nomic_anchor_pool.load_table")
    def test_different_paths_with_same_fingerprint_are_allowed(self, load_table):
        load_table.return_value = fake_table()
        with patch("pathlib.Path.is_file", return_value=True):
            selector = NomicPostMarkEmbedder(
                "/different/model/path",
                "/different/table/path.pt",
                tokenizer_path="/different/tokenizer/path",
                device="cpu",
            )
        self.assertTrue(selector.config_consistent)
        self.assertTrue(selector.eligible_for_aggregate)
        result = selector.select("one two three four five six seven eight nine", top_k=1)
        self.assertIsInstance(result, SelectionResult)
        self.assertEqual(result.words, ["alpha"])

    @patch("postmark.nomic_embedder.NomicTextEncoder", FakeEncoder)
    @patch("postmark.build_nomic_anchor_pool.load_table")
    def test_fingerprint_mismatch_fails_by_default(self, load_table):
        load_table.return_value = fake_table(model_digest="f" * 64)
        with patch("pathlib.Path.is_file", return_value=True):
            with self.assertRaises(ResourceMismatchError):
                NomicPostMarkEmbedder(
                    "/model",
                    "/table.pt",
                    tokenizer_path="/tokenizer",
                    device="cpu",
                )

    @patch("postmark.nomic_embedder.NomicTextEncoder", FakeEncoder)
    @patch("postmark.build_nomic_anchor_pool.load_table")
    def test_mismatch_override_is_diagnostic_only(self, load_table):
        load_table.return_value = fake_table(model_digest="f" * 64)
        with patch("pathlib.Path.is_file", return_value=True):
            selector = NomicPostMarkEmbedder(
                "/model",
                "/table.pt",
                tokenizer_path="/tokenizer",
                device="cpu",
                allow_resource_mismatch=True,
            )
        self.assertFalse(selector.config_consistent)
        self.assertFalse(selector.eligible_for_aggregate)

    @patch("postmark.nomic_embedder.NomicTextEncoder", FakeEncoder)
    @patch("postmark.build_nomic_anchor_pool.load_table")
    def test_k_zero_does_not_encode(self, load_table):
        load_table.return_value = fake_table()
        with patch("pathlib.Path.is_file", return_value=True):
            selector = NomicPostMarkEmbedder(
                "/model",
                "/table.pt",
                tokenizer_path="/tokenizer",
                device="cpu",
            )
        result = selector.select("short", ratio=0.12)
        self.assertEqual(result.words, [])
        self.assertEqual(result.k, 0)
        self.assertEqual(selector.encoder.encode_calls, 0)

    @patch("postmark.nomic_embedder.NomicTextEncoder", FakeEncoder)
    @patch("postmark.build_nomic_anchor_pool.load_table")
    def test_selection_hash_binds_ratio_and_mode(self, load_table):
        baseline = self.make_selector(load_table, ratio=0.12)
        changed_ratio = self.make_selector(load_table, ratio=0.2)
        changed_mode = self.make_selector(load_table, selection_mode="anchor_only")
        self.assertNotEqual(
            baseline.selection_config_sha256,
            changed_ratio.selection_config_sha256,
        )
        self.assertNotEqual(
            baseline.selection_config_sha256,
            changed_mode.selection_config_sha256,
        )
        self.assertFalse(changed_mode.paper_method_compatible)


if __name__ == "__main__":
    unittest.main()
