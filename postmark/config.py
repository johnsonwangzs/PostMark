"""Strict configuration schema for the local PostMark pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from .common import ConfigurationError, config_sha256, load_json_object


CONFIG_SCHEMA_VERSION = 1
T = TypeVar("T")


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ConfigurationError(f"{context} is missing fields: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"{context} has unknown fields: {sorted(unknown)}")


def _require_type(value: Any, expected_type: type[T], context: str) -> T:
    if expected_type in (int, float) and isinstance(value, bool):
        raise ConfigurationError(f"{context} must be {expected_type.__name__}")
    if not isinstance(value, expected_type):
        raise ConfigurationError(f"{context} must be {expected_type.__name__}")
    return value


def _require_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{context} must be numeric")
    return float(value)


@dataclass(frozen=True)
class LocalPathsConfig:
    inserter: str
    embedder: str
    embedder_tokenizer: str
    anchor_corpus: str
    candidate_words_legacy: str
    candidate_words: str
    nomic_table: str
    insertion_prompt: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LocalPathsConfig":
        fields = set(cls.__dataclass_fields__)
        _require_keys(value, fields, "paths")
        for key in fields:
            path = _require_type(value[key], str, f"paths.{key}")
            if not path:
                raise ConfigurationError(f"paths.{key} cannot be empty")
        return cls(**value)

    def resolved(self, project_root: str | Path) -> dict[str, Path]:
        root = Path(project_root)
        result: dict[str, Path] = {}
        for key, raw_path in asdict(self).items():
            path = Path(raw_path).expanduser()
            result[key] = path if path.is_absolute() else root / path
        return result


@dataclass(frozen=True)
class EmbeddingConfig:
    pooling: str
    normalization: str
    max_length: int
    task_prefix: str
    local_files_only: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "embedding")
        if value["pooling"] != "mean":
            raise ConfigurationError("embedding.pooling must be 'mean'")
        if value["normalization"] != "l2":
            raise ConfigurationError("embedding.normalization must be 'l2'")
        max_length = _require_type(value["max_length"], int, "embedding.max_length")
        if max_length < 1:
            raise ConfigurationError("embedding.max_length must be positive")
        _require_type(value["task_prefix"], str, "embedding.task_prefix")
        local_only = _require_type(
            value["local_files_only"], bool, "embedding.local_files_only"
        )
        if not local_only:
            raise ConfigurationError("PostMark-Local requires local_files_only=true")
        return cls(**value)


@dataclass(frozen=True)
class AnchorConfig:
    text_field: str
    chunk_words: int
    num_anchor_chunks: int
    chunking_algorithm: str
    seed: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnchorConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "anchor")
        if not _require_type(value["text_field"], str, "anchor.text_field"):
            raise ConfigurationError("anchor.text_field cannot be empty")
        for key in ("chunk_words", "num_anchor_chunks"):
            number = _require_type(value[key], int, f"anchor.{key}")
            if number < 1:
                raise ConfigurationError(f"anchor.{key} must be positive")
        if value["chunking_algorithm"] not in {"sentbound_v1", "nonoverlap_v1"}:
            raise ConfigurationError("Unsupported anchor.chunking_algorithm")
        _require_type(value["seed"], int, "anchor.seed")
        return cls(**value)


@dataclass(frozen=True)
class SelectionConfig:
    ratio: float
    prefilter_multiplier: int
    word_count_rule: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SelectionConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "selection")
        ratio = _require_number(value["ratio"], "selection.ratio")
        if not 0 <= ratio <= 1:
            raise ConfigurationError("selection.ratio must be in [0, 1]")
        multiplier = _require_type(
            value["prefilter_multiplier"], int, "selection.prefilter_multiplier"
        )
        if multiplier < 1:
            raise ConfigurationError("selection.prefilter_multiplier must be positive")
        if value["word_count_rule"] not in {
            "whitespace_split_floor",
            "portable_round_clamp",
        }:
            raise ConfigurationError("Unsupported selection.word_count_rule")
        return cls(
            ratio=ratio,
            prefilter_multiplier=multiplier,
            word_count_rule=value["word_count_rule"],
        )


@dataclass(frozen=True)
class InsertionConfig:
    iterate: str
    group_size: int
    min_group_presence: float
    max_insert_attempts: int
    max_new_tokens: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InsertionConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "insertion")
        if value["iterate"] != "v2":
            raise ConfigurationError("insertion.iterate must be 'v2'")
        for key in ("group_size", "max_insert_attempts", "max_new_tokens"):
            number = _require_type(value[key], int, f"insertion.{key}")
            if number < 1:
                raise ConfigurationError(f"insertion.{key} must be positive")
        presence = _require_number(
            value["min_group_presence"], "insertion.min_group_presence"
        )
        if not 0 <= presence <= 1:
            raise ConfigurationError("insertion.min_group_presence must be in [0, 1]")
        return cls(
            iterate=value["iterate"],
            group_size=value["group_size"],
            min_group_presence=presence,
            max_insert_attempts=value["max_insert_attempts"],
            max_new_tokens=value["max_new_tokens"],
        )


@dataclass(frozen=True)
class DetectorConfig:
    presence_mode: str
    spacy_model: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectorConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "detector")
        if value["presence_mode"] not in {"exact_lemma", "nomic_fuzzy"}:
            raise ConfigurationError(
                "Portable detector.presence_mode must be exact_lemma or nomic_fuzzy"
            )
        if not _require_type(value["spacy_model"], str, "detector.spacy_model"):
            raise ConfigurationError("detector.spacy_model cannot be empty")
        return cls(**value)


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    offline: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "runtime")
        _require_type(value["seed"], int, "runtime.seed")
        offline = _require_type(value["offline"], bool, "runtime.offline")
        if not offline:
            raise ConfigurationError("PostMark-Local requires runtime.offline=true")
        return cls(**value)


@dataclass(frozen=True)
class PostMarkConfig:
    schema_version: int
    implementation_profile: str
    detector_profile: str
    selection_mode: str
    paths: LocalPathsConfig
    embedding: EmbeddingConfig
    anchor: AnchorConfig
    selection: SelectionConfig
    insertion: InsertionConfig
    detector: DetectorConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PostMarkConfig":
        _require_keys(value, set(cls.__dataclass_fields__), "config")
        if value["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported config schema version: {value['schema_version']}"
            )
        if value["implementation_profile"] not in {"compat", "portable"}:
            raise ConfigurationError("Unsupported implementation_profile")
        if value["detector_profile"] != "portable":
            raise ConfigurationError("This configuration schema currently supports portable detector")
        if value["selection_mode"] not in {
            "official_two_stage",
            "anchor_only",
            "direct_word",
        }:
            raise ConfigurationError("Unsupported selection_mode")
        nested_types = {
            "paths": LocalPathsConfig,
            "embedding": EmbeddingConfig,
            "anchor": AnchorConfig,
            "selection": SelectionConfig,
            "insertion": InsertionConfig,
            "detector": DetectorConfig,
            "runtime": RuntimeConfig,
        }
        for key in nested_types:
            _require_type(value[key], dict, key)
        return cls(
            schema_version=value["schema_version"],
            implementation_profile=value["implementation_profile"],
            detector_profile=value["detector_profile"],
            selection_mode=value["selection_mode"],
            paths=LocalPathsConfig.from_dict(value["paths"]),
            embedding=EmbeddingConfig.from_dict(value["embedding"]),
            anchor=AnchorConfig.from_dict(value["anchor"]),
            selection=SelectionConfig.from_dict(value["selection"]),
            insertion=InsertionConfig.from_dict(value["insertion"]),
            detector=DetectorConfig.from_dict(value["detector"]),
            runtime=RuntimeConfig.from_dict(value["runtime"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PostMarkConfig":
        return cls.from_dict(load_json_object(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return config_sha256(self.to_dict())

    def validate_local_resources(self, project_root: str | Path) -> dict[str, Path]:
        resolved = self.paths.resolved(project_root)
        expected_directories = {"inserter", "embedder", "embedder_tokenizer"}
        for key, path in resolved.items():
            if key in expected_directories:
                valid = path.is_dir()
                expected = "directory"
            else:
                valid = path.is_file()
                expected = "file"
            if not valid:
                raise ConfigurationError(
                    f"paths.{key} must reference an existing local {expected}: {path}"
                )
        return resolved
