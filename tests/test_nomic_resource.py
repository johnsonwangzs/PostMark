import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from postmark.build_nomic_anchor_pool import (
    build_table_data,
    collect_unique_chunks,
    iter_text_chunks,
    load_table,
    save_table,
)
from postmark.common import ResourceError
from postmark.resources import tensor_bundle_sha256


class FakeEncoder:
    embedding_dim = 4

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = torch.tensor([digest[index] + 1 for index in range(4)], dtype=torch.float32)
            rows.append(F.normalize(row, dim=0))
        return torch.stack(rows)


def build_fixture_table(seed: int = 42):
    return build_table_data(
        candidate_words=["alpha", "beta", "gamma"],
        candidate_resource_sha256="a" * 64,
        chunks=[
            "first anchor sentence",
            "second anchor sentence",
            "third anchor sentence",
            "fourth anchor sentence",
            "fifth anchor sentence",
        ],
        encoder=FakeEncoder(),
        seed=seed,
        implementation_profile="compat",
        selection_mode="official_two_stage",
        prefilter_multiplier=3,
        mapping_algorithm_version=1,
        source_metadata={"corpus_sha256": "b" * 64, "num_anchor_chunks": 5},
        embedder_metadata={
            "fingerprint": {"sha256": "c" * 64},
            "tokenizer_fingerprint": {"sha256": "d" * 64},
            "embedding_dim": 4,
        },
    )


class ChunkingTests(unittest.TestCase):
    def test_sentbound_prefers_last_boundary_in_window(self):
        text = "one two three four. five six seven eight nine ten"
        chunks = list(
            iter_text_chunks(
                text,
                chunk_words=6,
                min_chunk_words=4,
                algorithm="sentbound_v1",
            )
        )
        self.assertEqual(chunks, ["one two three four.", "five six seven eight nine ten"])

    def test_collection_is_stable_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            records = [
                {"text": "one two three four five six"},
                {"text": "one two three four five six"},
                {"text": "seven eight nine ten eleven twelve"},
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = collect_unique_chunks(
                path,
                text_field="text",
                num_anchor_chunks=2,
                chunk_words=6,
                min_chunk_words=4,
                chunking_algorithm="sentbound_v1",
            )
            self.assertEqual(result.records_scanned, 3)
            self.assertEqual(result.duplicate_chunks, 1)
            self.assertEqual(len(result.chunks), 2)


class NomicTableTests(unittest.TestCase):
    def test_mapping_is_seeded(self):
        first = build_fixture_table(seed=42)
        repeated = build_fixture_table(seed=42)
        different = build_fixture_table(seed=43)
        self.assertEqual(first["selected_indices"], repeated["selected_indices"])
        self.assertTrue(
            torch.equal(first["anchor_embeddings"], repeated["anchor_embeddings"])
        )
        self.assertNotEqual(first["selected_indices"], different["selected_indices"])

    def test_table_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.pt"
            original = build_fixture_table()
            manifest = save_table(original, path)
            loaded, loaded_manifest = load_table(path)
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(original["candidate_words"], loaded["candidate_words"])
            self.assertTrue(
                torch.equal(original["anchor_embeddings"], loaded["anchor_embeddings"])
            )
            self.assertTrue(
                torch.equal(
                    original["candidate_word_embeddings"],
                    loaded["candidate_word_embeddings"],
                )
            )

    def test_artifact_tampering_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.pt"
            save_table(build_fixture_table(), path)
            with path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(ResourceError):
                load_table(path)

    def test_inconsistent_candidate_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            table = build_fixture_table()
            table["candidate_words_sha256"] = "0" * 64
            with self.assertRaises(ResourceError):
                save_table(table, Path(directory) / "table.pt")

    def test_tensor_hash_binds_name_dtype_shape_and_bytes(self):
        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        baseline = tensor_bundle_sha256({"version": 1}, {"values": tensor})
        self.assertNotEqual(
            baseline,
            tensor_bundle_sha256({"version": 1}, {"other": tensor}),
        )
        self.assertNotEqual(
            baseline,
            tensor_bundle_sha256({"version": 1}, {"values": tensor.reshape(2, 1)}),
        )
        self.assertNotEqual(
            baseline,
            tensor_bundle_sha256({"version": 1}, {"values": tensor.to(torch.float64)}),
        )


if __name__ == "__main__":
    unittest.main()
