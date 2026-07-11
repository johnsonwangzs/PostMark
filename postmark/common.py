"""Deterministic, offline-safe utilities shared by PostMark-Local."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class PostMarkError(Exception):
    """Base exception for expected PostMark-Local failures."""


class ConfigurationError(PostMarkError):
    """Raised when a configuration is missing or internally inconsistent."""


class JsonlError(PostMarkError):
    """Raised when a JSONL input violates the pipeline contract."""


class DuplicateIdError(JsonlError):
    """Raised when a supposedly unique sample ID is repeated."""


class ResourceError(PostMarkError):
    """Raised when a local resource cannot be read or validated."""


class ResourceMismatchError(ResourceError):
    """Raised when a resource or configuration fingerprint does not match."""


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically and reject non-finite numbers."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Value is not canonical JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_dumps(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    resource_path = Path(path)
    if not resource_path.is_file():
        raise ResourceError(f"Expected a local file: {resource_path}")

    digest = hashlib.sha256()
    with resource_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_loads(payload: str, *, source: str = "JSON") -> Any:
    """Load JSON while rejecting duplicate object keys and NaN/Infinity."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JsonlError(f"Duplicate key {key!r} in {source}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise JsonlError(f"Non-finite number {value!r} in {source}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except JsonlError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise JsonlError(f"Invalid {source}: {exc}") from exc


def load_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        payload = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration {config_path}: {exc}") from exc

    try:
        value = strict_json_loads(payload, source=str(config_path))
    except JsonlError as exc:
        raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration must be a JSON object: {config_path}")
    return value


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_json(dict(config))


def require_sample_id(record: Mapping[str, Any], id_field: str = "id") -> str:
    if id_field not in record:
        raise JsonlError(f"Missing sample ID field {id_field!r}")
    value = record[id_field]
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise JsonlError(f"Sample ID {id_field!r} must be a string or integer")
    sample_id = str(value)
    if not sample_id:
        raise JsonlError(f"Sample ID {id_field!r} cannot be empty")
    return sample_id


def stable_content_id(value: Any, *, prefix: str = "sample") -> str:
    return f"{prefix}-{sha256_json(value)[:24]}"


def derive_sample_seed(base_seed: int, sample_id: str, *, attempt: int = 0) -> int:
    if attempt < 0:
        raise ConfigurationError("attempt must be non-negative")
    payload = canonical_json_bytes(
        {"base_seed": int(base_seed), "sample_id": str(sample_id), "attempt": attempt}
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def set_global_seed(seed: int) -> None:
    """Seed installed numerical libraries without importing them at module load."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_word_count(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(text.split())


def iter_jsonl(
    path: str | os.PathLike[str],
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ConfigurationError("limit must be non-negative")

    input_path = Path(path)
    if not input_path.is_file():
        raise JsonlError(f"JSONL file does not exist: {input_path}")

    yielded = 0
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if limit is not None and yielded >= limit:
                break
            if not raw_line.strip():
                raise JsonlError(f"Blank line at {input_path}:{line_number}")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise JsonlError(f"Invalid UTF-8 at {input_path}:{line_number}") from exc
            value = strict_json_loads(line, source=f"{input_path}:{line_number}")
            if not isinstance(value, dict):
                raise JsonlError(f"Record must be an object at {input_path}:{line_number}")
            yield value
            yielded += 1


def load_jsonl(
    path: str | os.PathLike[str],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, limit=limit))


def index_records_by_id(
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "id",
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = require_sample_id(record, id_field)
        if sample_id in indexed:
            raise DuplicateIdError(f"Duplicate sample ID {sample_id!r}")
        indexed[sample_id] = record
    return indexed


def append_jsonl_record(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one durable canonical JSONL record."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size:
        with output_path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                raise JsonlError(
                    f"Refusing to append to JSONL without a final newline: {output_path}"
                )

    payload = canonical_json_bytes(dict(record)) + b"\n"
    with output_path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def recover_truncated_jsonl_tail(path: str | os.PathLike[str]) -> Path | None:
    """Back up and remove only an invalid, non-newline-terminated final record."""

    input_path = Path(path)
    if not input_path.is_file():
        raise JsonlError(f"JSONL file does not exist: {input_path}")

    last_good_offset = 0
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            end_offset = handle.tell()
            try:
                decoded = raw_line.decode("utf-8")
                value = strict_json_loads(decoded, source=f"{input_path}:{line_number}")
                if not isinstance(value, dict):
                    raise JsonlError(
                        f"Record must be an object at {input_path}:{line_number}"
                    )
            except (JsonlError, UnicodeDecodeError) as exc:
                if raw_line.endswith(b"\n"):
                    raise JsonlError(
                        f"Cannot repair non-tail corruption at {input_path}:{line_number}"
                    ) from exc
                backup_path = input_path.with_name(input_path.name + ".corrupt.bak")
                suffix = 1
                while backup_path.exists():
                    backup_path = input_path.with_name(
                        input_path.name + f".corrupt.bak.{suffix}"
                    )
                    suffix += 1
                shutil.copy2(input_path, backup_path)
                with input_path.open("r+b") as output:
                    output.truncate(last_good_offset)
                    output.flush()
                    os.fsync(output.fileno())
                return backup_path
            last_good_offset = end_offset
    return None


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    pretty: bool = True,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    else:
        payload = canonical_json_dumps(value) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ResourceError(f"Cannot atomically write {output_path}: {exc}") from exc
