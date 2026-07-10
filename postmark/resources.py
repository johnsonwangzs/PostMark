"""Versioned resource manifests and path-independent local fingerprints."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    ConfigurationError,
    ResourceError,
    ResourceMismatchError,
    atomic_write_json,
    load_json_object,
    sha256_file,
)


MANIFEST_SCHEMA_VERSION = 1


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
