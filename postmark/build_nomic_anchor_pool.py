"""Build the fixed local Nomic anchor and candidate-word embedding table."""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from .common import (
    ConfigurationError,
    JsonlError,
    ResourceError,
    ResourceMismatchError,
    iter_jsonl,
    sha256_file,
    sha256_json,
)
from .nomic_embedder import NomicTextEncoder
from .resources import (
    ResourceManifest,
    load_candidate_words,
    load_manifest,
    tensor_bundle_sha256,
    validate_candidate_words,
    verify_manifest,
    write_manifest,
)


TABLE_VERSION = 2


class TextEncoder(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def encode_texts(self, texts: list[str]) -> torch.Tensor: ...


@dataclass(frozen=True)
class ChunkCollection:
    chunks: list[str]
    records_scanned: int
    rejected_short: int
    rejected_language: int
    duplicate_chunks: int


def _is_english_like(text: str, *, minimum_ascii_ratio: float = 0.9) -> bool:
    alphabetic = [character for character in text if character.isalpha()]
    if not alphabetic:
        return False
    ascii_letters = sum(character.isascii() for character in alphabetic)
    return ascii_letters / len(alphabetic) >= minimum_ascii_ratio


def iter_text_chunks(
    text: str,
    *,
    chunk_words: int,
    algorithm: str = "sentbound_v1",
    min_chunk_words: int | None = None,
) -> Iterable[str]:
    if chunk_words < 1:
        raise ConfigurationError("chunk_words must be positive")
    minimum = max(1, int(chunk_words * 0.8)) if min_chunk_words is None else min_chunk_words
    if minimum < 1 or minimum > chunk_words:
        raise ConfigurationError("min_chunk_words must be in [1, chunk_words]")
    if algorithm not in {"sentbound_v1", "nonoverlap_v1"}:
        raise ConfigurationError(f"Unsupported chunking algorithm: {algorithm}")

    words = text.split()
    start = 0
    while len(words) - start >= minimum:
        target_end = min(start + chunk_words, len(words))
        end = target_end
        if algorithm == "sentbound_v1" and target_end < len(words):
            boundary_start = start + minimum
            for index in range(target_end - 1, boundary_start - 2, -1):
                token = words[index].rstrip("\"')]}")
                if token.endswith((".", "!", "?")):
                    end = index + 1
                    break
        chunk = " ".join(words[start:end])
        if chunk:
            yield chunk
        start = end


def collect_unique_chunks(
    input_path: str | Path,
    *,
    text_field: str,
    num_anchor_chunks: int,
    chunk_words: int,
    chunking_algorithm: str,
    min_chunk_words: int | None = None,
) -> ChunkCollection:
    if num_anchor_chunks < 1:
        raise ConfigurationError("num_anchor_chunks must be positive")
    chunks: list[str] = []
    seen_hashes: set[str] = set()
    records_scanned = 0
    rejected_short = 0
    rejected_language = 0
    duplicate_chunks = 0

    for record in iter_jsonl(input_path):
        records_scanned += 1
        if text_field not in record or not isinstance(record[text_field], str):
            raise JsonlError(
                f"Record {records_scanned} is missing string field {text_field!r}"
            )
        emitted = False
        for chunk in iter_text_chunks(
            record[text_field],
            chunk_words=chunk_words,
            algorithm=chunking_algorithm,
            min_chunk_words=min_chunk_words,
        ):
            emitted = True
            if not _is_english_like(chunk):
                rejected_language += 1
                continue
            chunk_hash = sha256_json(chunk)
            if chunk_hash in seen_hashes:
                duplicate_chunks += 1
                continue
            seen_hashes.add(chunk_hash)
            chunks.append(chunk)
            if len(chunks) == num_anchor_chunks:
                return ChunkCollection(
                    chunks=chunks,
                    records_scanned=records_scanned,
                    rejected_short=rejected_short,
                    rejected_language=rejected_language,
                    duplicate_chunks=duplicate_chunks,
                )
        if not emitted:
            rejected_short += 1
    raise ResourceError(
        f"Corpus yielded only {len(chunks)} unique chunks; required {num_anchor_chunks}"
    )


def _validate_embeddings(name: str, tensor: torch.Tensor, expected_rows: int) -> None:
    if tensor.dtype != torch.float32 or tensor.ndim != 2:
        raise ResourceError(f"{name} must be a 2D float32 tensor")
    if tensor.shape[0] != expected_rows:
        raise ResourceError(f"{name} row count does not match candidate words")
    if not torch.isfinite(tensor).all():
        raise ResourceError(f"{name} contains NaN or Inf")
    norms = torch.linalg.vector_norm(tensor, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ResourceError(f"{name} rows must be L2 normalized")


def build_table_data(
    *,
    candidate_words: list[str],
    candidate_resource_sha256: str,
    chunks: list[str],
    encoder: TextEncoder,
    seed: int,
    implementation_profile: str,
    selection_mode: str,
    prefilter_multiplier: int,
    mapping_algorithm_version: int,
    source_metadata: dict[str, Any],
    embedder_metadata: dict[str, Any],
) -> dict[str, Any]:
    validate_candidate_words(candidate_words)
    if len(chunks) < len(candidate_words):
        raise ConfigurationError("Anchor chunk count must be at least candidate word count")
    if prefilter_multiplier < 1 or mapping_algorithm_version < 1:
        raise ConfigurationError(
            "prefilter_multiplier and mapping_algorithm_version must be positive"
        )
    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(chunks)), len(candidate_words))
    permutation = list(range(len(candidate_words)))
    rng.shuffle(permutation)
    mapped_indices = [sampled_indices[index] for index in permutation]
    selected_chunks = [chunks[index] for index in mapped_indices]

    anchor_embeddings = encoder.encode_texts(selected_chunks).cpu().to(torch.float32)
    candidate_embeddings = encoder.encode_texts(candidate_words).cpu().to(torch.float32)
    _validate_embeddings("anchor_embeddings", anchor_embeddings, len(candidate_words))
    _validate_embeddings(
        "candidate_word_embeddings", candidate_embeddings, len(candidate_words)
    )
    if anchor_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ResourceError("Anchor and candidate embedding dimensions differ")

    return {
        "version": TABLE_VERSION,
        "implementation_profile": implementation_profile,
        "selection_mode": selection_mode,
        "candidate_words": candidate_words,
        "anchor_embeddings": anchor_embeddings,
        "candidate_word_embeddings": candidate_embeddings,
        "seed": seed,
        "prefilter_multiplier": prefilter_multiplier,
        "mapping_algorithm_version": mapping_algorithm_version,
        "candidate_resource_sha256": candidate_resource_sha256,
        "candidate_words_sha256": sha256_json(candidate_words),
        "selected_indices": mapped_indices,
        "selected_chunks_sha256": sha256_json(selected_chunks),
        "embedder": embedder_metadata,
        "source": source_metadata,
    }


def _table_parts(table: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    required_tensors = {"anchor_embeddings", "candidate_word_embeddings"}
    missing = required_tensors - table.keys()
    if missing:
        raise ResourceError(f"Nomic table is missing tensors: {sorted(missing)}")
    tensors = {name: table[name] for name in required_tensors}
    metadata = {key: value for key, value in table.items() if key not in required_tensors}
    return metadata, tensors


def validate_table(table: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metadata, tensors = _table_parts(table)
    if table.get("version") != TABLE_VERSION:
        raise ResourceError(f"Unsupported Nomic table version: {table.get('version')}")
    if table.get("implementation_profile") != "compat":
        raise ResourceError("Nomic table implementation_profile must be compat")
    if table.get("selection_mode") != "official_two_stage":
        raise ResourceError("Nomic table selection_mode must be official_two_stage")

    candidate_words = table.get("candidate_words")
    if not isinstance(candidate_words, list):
        raise ResourceError("Nomic table candidate_words must be a list")
    try:
        validate_candidate_words(candidate_words)
    except ConfigurationError as exc:
        raise ResourceError(f"Invalid Nomic table candidate words: {exc}") from exc
    if table.get("candidate_words_sha256") != sha256_json(candidate_words):
        raise ResourceMismatchError("Nomic table candidate words hash mismatch")

    selected_indices = table.get("selected_indices")
    if (
        not isinstance(selected_indices, list)
        or len(selected_indices) != len(candidate_words)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in selected_indices)
        or len(set(selected_indices)) != len(selected_indices)
    ):
        raise ResourceError("Nomic table selected_indices must be unique integer indices")
    source = table.get("source")
    if not isinstance(source, dict):
        raise ResourceError("Nomic table source metadata must be an object")
    num_anchor_chunks = source.get("num_anchor_chunks")
    if (
        isinstance(num_anchor_chunks, bool)
        or not isinstance(num_anchor_chunks, int)
        or num_anchor_chunks < len(candidate_words)
        or any(index < 0 or index >= num_anchor_chunks for index in selected_indices)
    ):
        raise ResourceError("Nomic table selected indices exceed the anchor pool")

    _validate_embeddings("anchor_embeddings", tensors["anchor_embeddings"], len(candidate_words))
    _validate_embeddings(
        "candidate_word_embeddings",
        tensors["candidate_word_embeddings"],
        len(candidate_words),
    )
    if tensors["anchor_embeddings"].shape[1] != tensors["candidate_word_embeddings"].shape[1]:
        raise ResourceError("Anchor and candidate embedding dimensions differ")
    embedder = table.get("embedder")
    if not isinstance(embedder, dict) or embedder.get("embedding_dim") != tensors[
        "anchor_embeddings"
    ].shape[1]:
        raise ResourceError("Nomic table embedder dimension is inconsistent")
    return metadata, tensors


def table_manifest_path(table_path: str | Path) -> Path:
    return Path(table_path).with_suffix(".manifest.json")


def save_table(table: dict[str, Any], output_path: str | Path) -> ResourceManifest:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata, tensors = validate_table(table)
    content_hash = tensor_bundle_sha256(metadata, tensors)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(table, temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, RuntimeError, TypeError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ResourceError(f"Cannot save Nomic table {output}: {exc}") from exc

    manifest = ResourceManifest(
        resource_type="postmark_nomic_table",
        resource_version=TABLE_VERSION,
        content_sha256=content_hash,
        metadata={
            "artifact_sha256": sha256_file(output),
            "candidate_words_sha256": table["candidate_words_sha256"],
            "selected_chunks_sha256": table["selected_chunks_sha256"],
            "embedding_dim": tensors["anchor_embeddings"].shape[1],
            "num_candidate_words": len(table["candidate_words"]),
        },
        fingerprints={
            "embedder": table["embedder"]["fingerprint"],
            "tokenizer": table["embedder"]["tokenizer_fingerprint"],
            "corpus": table["source"]["corpus_sha256"],
            "candidate_resource": table["candidate_resource_sha256"],
        },
    )
    write_manifest(table_manifest_path(output), manifest)
    return manifest


def load_table(path: str | Path) -> tuple[dict[str, Any], ResourceManifest]:
    table_path = Path(path)
    manifest = load_manifest(table_manifest_path(table_path))
    artifact_hash = sha256_file(table_path)
    if manifest.metadata.get("artifact_sha256") != artifact_hash:
        raise ResourceError("Nomic table artifact hash mismatch")
    try:
        table = torch.load(table_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ResourceError(f"Cannot load Nomic table {table_path}: {exc}") from exc
    if not isinstance(table, dict):
        raise ResourceError("Nomic table must contain a dictionary")
    metadata, tensors = validate_table(table)
    verify_manifest(
        manifest,
        computed_content_sha256=tensor_bundle_sha256(metadata, tensors),
        expected_resource_type="postmark_nomic_table",
        expected_resource_version=TABLE_VERSION,
    )
    cross_checks = {
        "candidate_words_sha256": table["candidate_words_sha256"],
        "selected_chunks_sha256": table["selected_chunks_sha256"],
        "embedding_dim": tensors["anchor_embeddings"].shape[1],
        "num_candidate_words": len(table["candidate_words"]),
    }
    for key, expected in cross_checks.items():
        if manifest.metadata.get(key) != expected:
            raise ResourceMismatchError(f"Manifest metadata mismatch for {key}")
    expected_fingerprints = {
        "embedder": table["embedder"]["fingerprint"],
        "tokenizer": table["embedder"]["tokenizer_fingerprint"],
        "corpus": table["source"]["corpus_sha256"],
        "candidate_resource": table["candidate_resource_sha256"],
    }
    if manifest.fingerprints != expected_fingerprints:
        raise ResourceMismatchError("Manifest resource fingerprints do not match table")
    return table, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local PostMark Nomic table.")
    parser.add_argument("--implementation_profile", choices=("compat",), default="compat")
    parser.add_argument(
        "--selection_mode", choices=("official_two_stage",), default="official_two_stage"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--candidate_words_path", required=True)
    parser.add_argument("--embedder_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--corpus_revision", required=True)
    parser.add_argument("--corpus_sha256")
    parser.add_argument("--chunk_words", type=int, default=250)
    parser.add_argument("--num_anchor_chunks", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--chunking_algorithm", choices=("sentbound_v1",), default="sentbound_v1"
    )
    parser.add_argument("--mapping_algorithm_version", type=int, default=1)
    parser.add_argument("--pooling", choices=("mean",), default="mean")
    parser.add_argument("--normalization", choices=("l2",), default="l2")
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--prefilter_multiplier", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_resource = load_candidate_words(args.candidate_words_path)
    if candidate_resource.profile != "compat":
        raise ConfigurationError("Compat table requires a compat candidate resource")
    if args.num_anchor_chunks < len(candidate_resource.words):
        raise ConfigurationError(
            "num_anchor_chunks must be at least the number of candidate words"
        )
    corpus_hash = sha256_file(args.input_path)
    if args.corpus_sha256 is not None and args.corpus_sha256 != corpus_hash:
        raise ResourceError(
            f"Corpus hash mismatch: expected={args.corpus_sha256}, computed={corpus_hash}"
        )

    collection = collect_unique_chunks(
        args.input_path,
        text_field=args.text_field,
        num_anchor_chunks=args.num_anchor_chunks,
        chunk_words=args.chunk_words,
        chunking_algorithm=args.chunking_algorithm,
    )
    encoder = NomicTextEncoder(
        args.embedder_path,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        task_prefix=args.task_prefix,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=True,
    )
    source_metadata = {
        "corpus_name": Path(args.input_path).name,
        "corpus_revision": args.corpus_revision,
        "corpus_sha256": corpus_hash,
        "text_field": args.text_field,
        "chunk_words": args.chunk_words,
        "min_chunk_words": max(1, int(args.chunk_words * 0.8)),
        "num_anchor_chunks": args.num_anchor_chunks,
        "chunking_algorithm": args.chunking_algorithm,
        "records_scanned": collection.records_scanned,
        "rejected_short": collection.rejected_short,
        "rejected_language": collection.rejected_language,
        "duplicate_chunks": collection.duplicate_chunks,
    }
    embedder_metadata = {
        "fingerprint": encoder.model_fingerprint().to_dict(),
        "tokenizer_fingerprint": encoder.tokenizer_fingerprint().to_dict(),
        "pooling": args.pooling,
        "normalization": args.normalization,
        "max_length": args.max_length,
        "task_prefix": args.task_prefix,
        "embedding_dim": encoder.embedding_dim,
    }
    table = build_table_data(
        candidate_words=candidate_resource.words,
        candidate_resource_sha256=sha256_file(args.candidate_words_path),
        chunks=collection.chunks,
        encoder=encoder,
        seed=args.seed,
        implementation_profile=args.implementation_profile,
        selection_mode=args.selection_mode,
        prefilter_multiplier=args.prefilter_multiplier,
        mapping_algorithm_version=args.mapping_algorithm_version,
        source_metadata=source_metadata,
        embedder_metadata=embedder_metadata,
    )
    manifest = save_table(table, args.output_path)
    print(
        f"Wrote Nomic table with {len(candidate_resource.words)} words to "
        f"{args.output_path} (content_sha256={manifest.content_sha256})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
