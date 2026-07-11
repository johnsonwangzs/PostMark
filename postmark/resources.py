"""Versioned resource manifests and path-independent local fingerprints."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    ConfigurationError,
    ResourceError,
    ResourceMismatchError,
    atomic_write_json,
    canonical_json_bytes,
    load_json_object,
    sha256_file,
    sha256_json,
)


MANIFEST_SCHEMA_VERSION = 1
CANDIDATE_WORDS_VERSION = 2


@dataclass(frozen=True)
class PathFingerprint:
    sha256: str
    kind: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "kind": self.kind,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class ResourceManifest:
    resource_type: str
    resource_version: int
    content_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprints: dict[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.resource_type, str) or not self.resource_type:
            raise ConfigurationError("resource_type cannot be empty")
        if (
            isinstance(self.resource_version, bool)
            or not isinstance(self.resource_version, int)
            or self.resource_version < 1
        ):
            raise ConfigurationError("resource_version must be positive")
        if not isinstance(self.content_sha256, str) or len(self.content_sha256) != 64:
            raise ConfigurationError("content_sha256 must be a SHA-256 hex digest")
        try:
            int(self.content_sha256, 16)
        except ValueError as exc:
            raise ConfigurationError("content_sha256 must be hexadecimal") from exc
        if not isinstance(self.metadata, dict) or not isinstance(self.fingerprints, dict):
            raise ConfigurationError("Manifest metadata and fingerprints must be objects")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported manifest schema version: {self.schema_version}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resource_type": self.resource_type,
            "resource_version": self.resource_version,
            "content_sha256": self.content_sha256,
            "metadata": self.metadata,
            "fingerprints": self.fingerprints,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResourceManifest":
        expected_keys = {
            "schema_version",
            "resource_type",
            "resource_version",
            "content_sha256",
            "metadata",
            "fingerprints",
        }
        missing = expected_keys - value.keys()
        unknown = value.keys() - expected_keys
        if missing:
            raise ConfigurationError(f"Manifest is missing fields: {sorted(missing)}")
        if unknown:
            raise ConfigurationError(f"Manifest has unknown fields: {sorted(unknown)}")
        if not isinstance(value["metadata"], dict) or not isinstance(
            value["fingerprints"], dict
        ):
            raise ConfigurationError("Manifest metadata and fingerprints must be objects")
        return cls(
            schema_version=value["schema_version"],
            resource_type=value["resource_type"],
            resource_version=value["resource_version"],
            content_sha256=value["content_sha256"],
            metadata=value["metadata"],
            fingerprints=value["fingerprints"],
        )


@dataclass(frozen=True)
class CandidateWordsResource:
    profile: str
    source: str
    source_sha256: str
    words_sha256: str
    words: list[str]
    version: int = CANDIDATE_WORDS_VERSION

    def __post_init__(self) -> None:
        if self.version != CANDIDATE_WORDS_VERSION:
            raise ConfigurationError(
                f"Unsupported candidate words version: {self.version}"
            )
        if self.profile not in {"compat", "portable"}:
            raise ConfigurationError(f"Unsupported candidate words profile: {self.profile}")
        if not self.source:
            raise ConfigurationError("Candidate words source cannot be empty")
        for name, digest in (
            ("source_sha256", self.source_sha256),
            ("words_sha256", self.words_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ConfigurationError(f"{name} must be a SHA-256 hex digest")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be hexadecimal") from exc
        validate_candidate_words(self.words)
        computed = sha256_json(self.words)
        if computed != self.words_sha256:
            raise ResourceMismatchError(
                f"Candidate words hash mismatch: stored={self.words_sha256}, computed={computed}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "profile": self.profile,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "words_sha256": self.words_sha256,
            "words": self.words,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateWordsResource":
        expected = {
            "version",
            "profile",
            "source",
            "source_sha256",
            "words_sha256",
            "words",
        }
        missing = expected - value.keys()
        unknown = value.keys() - expected
        if missing:
            raise ConfigurationError(
                f"Candidate words resource is missing fields: {sorted(missing)}"
            )
        if unknown:
            raise ConfigurationError(
                f"Candidate words resource has unknown fields: {sorted(unknown)}"
            )
        if not isinstance(value["words"], list):
            raise ConfigurationError("Candidate words must be a list")
        return cls(**value)


def validate_candidate_words(words: list[str]) -> None:
    if not words:
        raise ConfigurationError("Candidate words cannot be empty")
    seen: set[str] = set()
    for index, word in enumerate(words):
        if not isinstance(word, str) or not word:
            raise ConfigurationError(
                f"Candidate word at index {index} must be a non-empty string"
            )
        if word in seen:
            raise ConfigurationError(f"Duplicate candidate word {word!r} at index {index}")
        seen.add(word)


def write_candidate_words(
    path: str | os.PathLike[str], resource: CandidateWordsResource
) -> None:
    atomic_write_json(path, resource.to_dict())


def load_candidate_words(path: str | os.PathLike[str]) -> CandidateWordsResource:
    return CandidateWordsResource.from_dict(load_json_object(path))


def fingerprint_path(path: str | os.PathLike[str]) -> PathFingerprint:
    """Fingerprint a file or directory without including its absolute path."""

    resource_path = Path(path)
    if resource_path.is_symlink():
        raise ResourceError(f"Resource symlinks are not supported: {resource_path}")
    if resource_path.is_file():
        return PathFingerprint(
            sha256=sha256_file(resource_path),
            kind="file",
            file_count=1,
            total_bytes=resource_path.stat().st_size,
        )
    if not resource_path.is_dir():
        raise ResourceError(f"Resource path does not exist: {resource_path}")

    entries = sorted(resource_path.rglob("*"))
    for candidate in entries:
        if candidate.is_symlink():
            raise ResourceError(f"Resource symlinks are not supported: {candidate}")
    files = [candidate for candidate in entries if candidate.is_file()]
    digest = hashlib.sha256()
    total_bytes = 0
    for candidate in files:
        relative_path = candidate.relative_to(resource_path).as_posix()
        size = candidate.stat().st_size
        file_hash = sha256_file(candidate)
        entry = f"{relative_path}\0{size}\0{file_hash}\n".encode("utf-8")
        digest.update(entry)
        total_bytes += size
    return PathFingerprint(
        sha256=digest.hexdigest(),
        kind="directory",
        file_count=len(files),
        total_bytes=total_bytes,
    )


def fingerprint_files(
    root: str | os.PathLike[str], relative_paths: list[str]
) -> PathFingerprint:
    """Fingerprint an explicit file set without including its absolute path."""

    resource_root = Path(root)
    if not resource_root.is_dir():
        raise ResourceError(f"Snapshot directory does not exist: {resource_root}")
    normalized_paths = sorted(set(relative_paths))
    if not normalized_paths:
        raise ResourceError("At least one snapshot file is required")

    digest = hashlib.sha256()
    total_bytes = 0
    for relative in normalized_paths:
        candidate = resource_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ResourceError(f"Snapshot file does not exist or is unsafe: {candidate}")
        size = candidate.stat().st_size
        file_hash = sha256_file(candidate)
        digest.update(f"{Path(relative).as_posix()}\0{size}\0{file_hash}\n".encode())
        total_bytes += size
    return PathFingerprint(
        sha256=digest.hexdigest(),
        kind="file_set",
        file_count=len(normalized_paths),
        total_bytes=total_bytes,
    )


def tensor_bundle_sha256(metadata: dict[str, Any], tensors: dict[str, Any]) -> str:
    """Hash canonical metadata plus named dense CPU tensor bytes."""

    digest = hashlib.sha256()
    metadata_bytes = canonical_json_bytes(metadata)
    digest.update(len(metadata_bytes).to_bytes(8, "big"))
    digest.update(metadata_bytes)

    try:
        import torch
    except ImportError as exc:
        raise ResourceError("PyTorch is required to hash tensor resources") from exc

    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise ResourceError(f"Tensor bundle entry {name!r} is not a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ResourceError(f"Tensor bundle entry {name!r} must be dense")
        normalized = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(normalized.dtype),
            "shape": list(normalized.shape),
            "byte_order": sys.byteorder,
        }
        header_bytes = canonical_json_bytes(header)
        raw_bytes = normalized.numpy().tobytes(order="C")
        digest.update(len(header_bytes).to_bytes(8, "big"))
        digest.update(header_bytes)
        digest.update(len(raw_bytes).to_bytes(8, "big"))
        digest.update(raw_bytes)
    return digest.hexdigest()


def write_manifest(path: str | os.PathLike[str], manifest: ResourceManifest) -> None:
    atomic_write_json(path, manifest.to_dict())


def load_manifest(path: str | os.PathLike[str]) -> ResourceManifest:
    try:
        value = load_json_object(path)
    except ConfigurationError as exc:
        raise ResourceError(f"Cannot load resource manifest {path}: {exc}") from exc
    return ResourceManifest.from_dict(value)


def verify_manifest(
    manifest: ResourceManifest,
    *,
    computed_content_sha256: str,
    expected_resource_type: str | None = None,
    expected_resource_version: int | None = None,
) -> None:
    if expected_resource_type is not None and manifest.resource_type != expected_resource_type:
        raise ResourceMismatchError(
            f"Expected resource type {expected_resource_type!r}, got {manifest.resource_type!r}"
        )
    if (
        expected_resource_version is not None
        and manifest.resource_version != expected_resource_version
    ):
        raise ResourceMismatchError(
            f"Expected resource version {expected_resource_version}, "
            f"got {manifest.resource_version}"
        )
    if manifest.content_sha256 != computed_content_sha256:
        raise ResourceMismatchError(
            "Resource content hash mismatch: "
            f"manifest={manifest.content_sha256}, computed={computed_content_sha256}"
        )


def build_file_manifest(
    path: str | os.PathLike[str],
    *,
    resource_type: str,
    resource_version: int = 1,
    metadata: dict[str, Any] | None = None,
    fingerprints: dict[str, Any] | None = None,
) -> ResourceManifest:
    return ResourceManifest(
        resource_type=resource_type,
        resource_version=resource_version,
        content_sha256=sha256_file(path),
        metadata={} if metadata is None else metadata,
        fingerprints={} if fingerprints is None else fingerprints,
    )
