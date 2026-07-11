"""CLI boundary for the local PostMark watermark pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Insert PostMark-Local watermarks using local resources."
    )
    parser.add_argument("--config", default="configs/postmark_portable.json")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--prompt_field")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--base_llm_path")
    parser.add_argument("--base_tokenizer_path")
    parser.add_argument("--inserter_path")
    parser.add_argument("--inserter_tokenizer_path")
    parser.add_argument("--embedder_path")
    parser.add_argument("--embedder_tokenizer_path")
    parser.add_argument("--table_path")
    parser.add_argument("--prompt_path")
    parser.add_argument(
        "--implementation_profile", choices=("compat", "portable"), default="compat"
    )
    parser.add_argument(
        "--selection_mode",
        choices=("official_two_stage", "anchor_only", "direct_word"),
        default="official_two_stage",
    )
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--min_watermark_words", type=int)
    parser.add_argument("--max_watermark_words", type=int)
    parser.add_argument("--iterate", choices=("v2",), default="v2")
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--min_group_presence", type=float)
    parser.add_argument("--max_insert_attempts", type=int)
    parser.add_argument("--retry_strategy", choices=("missing_words",), default="missing_words")
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--torch_dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--run_manifest_path")
    parser.add_argument("--allow_resource_mismatch", action="store_true")
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.error(
        "the watermark runtime is not available until the selector resource phase is complete"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
