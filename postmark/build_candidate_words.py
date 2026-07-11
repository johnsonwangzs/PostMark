"""Build a versioned candidate-word resource from trusted local inputs."""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

from .common import ConfigurationError, sha256_file, sha256_json
from .resources import (
    CandidateWordsResource,
    validate_candidate_words,
    write_candidate_words,
)


COMPAT_WORD_COUNT = 3266


class _StringListUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"Candidate word pickle may not load global {module}.{name}"
        )


def load_trusted_string_list(path: str | Path) -> list[str]:
    source_path = Path(path)
    if not source_path.is_file():
        raise ConfigurationError(f"Candidate word pickle does not exist: {source_path}")
    try:
        with source_path.open("rb") as handle:
            value = _StringListUnpickler(handle).load()
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ConfigurationError(f"Cannot read candidate word pickle: {exc}") from exc
    if not isinstance(value, list):
        raise ConfigurationError("Candidate word pickle must contain a list")
    validate_candidate_words(value)
    return value


def build_compat_candidate_words(
    legacy_pickle_path: str | Path,
    *,
    expected_word_count: int = COMPAT_WORD_COUNT,
) -> CandidateWordsResource:
    words = load_trusted_string_list(legacy_pickle_path)
    if len(words) != expected_word_count:
        raise ConfigurationError(
            f"Compat candidate list must contain {expected_word_count} words, got {len(words)}"
        )
    source_path = Path(legacy_pickle_path)
    return CandidateWordsResource(
        profile="compat",
        source=source_path.name,
        source_sha256=sha256_file(source_path),
        words_sha256=sha256_json(words),
        words=words,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the trusted PostMark candidate list to canonical JSON."
    )
    parser.add_argument(
        "--implementation_profile", choices=("compat",), default="compat"
    )
    parser.add_argument("--legacy_pickle_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_word_count", type=int, default=COMPAT_WORD_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resource = build_compat_candidate_words(
        args.legacy_pickle_path,
        expected_word_count=args.expected_word_count,
    )
    write_candidate_words(args.output_path, resource)
    print(
        f"Wrote {len(resource.words)} {resource.profile} candidate words to "
        f"{args.output_path} (words_sha256={resource.words_sha256})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
