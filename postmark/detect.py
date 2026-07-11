"""CLI boundary for blind PostMark-Local detection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect PostMark-Local watermarks using local keyed resources."
    )
    parser.add_argument("--config", default="configs/postmark_portable.json")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--table_path")
    parser.add_argument("--embedder_path")
    parser.add_argument("--embedder_tokenizer_path")
    parser.add_argument(
        "--implementation_profile", choices=("compat", "portable"), default="compat"
    )
    parser.add_argument(
        "--detector_profile", choices=("portable",), default="portable"
    )
    parser.add_argument(
        "--selection_mode",
        choices=("official_two_stage", "anchor_only", "direct_word"),
        default="official_two_stage",
    )
    parser.add_argument(
        "--presence_mode", choices=("exact_lemma", "nomic_fuzzy"), default="exact_lemma"
    )
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--negative_field", default="text1")
    parser.add_argument("--positive_field", default="text2")
    parser.add_argument("--calibration_path")
    parser.add_argument("--calibration_text_field", default="text")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_resource_mismatch", action="store_true")
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.error(
        "the detector runtime is not available until the selector resource phase is complete"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
