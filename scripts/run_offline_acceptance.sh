#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export POSTMARK_BLOCK_NETWORK=1

output_dir="${1:-/tmp/postmark-local-acceptance}"
mkdir -p "${output_dir}"

python -m postmark.watermark \
  --input_path data/opengen.jsonl \
  --output_path "${output_dir}/watermarked.jsonl" \
  --text_field prefix \
  --limit 3 \
  --ratio 0.005 \
  --min_watermark_words 2 \
  --max_watermark_words 2 \
  --group_size 2 \
  --min_group_presence 0.5 \
  --max_insert_attempts 1 \
  --max_new_tokens 384 \
  --overwrite

python -m postmark.detect \
  --input_path "${output_dir}/watermarked.jsonl" \
  --output_path "${output_dir}/scored.jsonl" \
  --paired \
  --negative_field text1 \
  --positive_field text2 \
  --calibration_path tests/fixtures/portable_calibration_smoke.jsonl \
  --calibration_text_field text \
  --ratio 0.005 \
  --min_watermark_words 2 \
  --max_watermark_words 2 \
  --presence_mode exact_lemma \
  --bootstrap_samples 200 \
  --bootstrap_seed 42 \
  --overwrite

python -m postmark.quality \
  --input_path "${output_dir}/watermarked.jsonl" \
  --output_path "${output_dir}/quality.json" \
  --sample_output_path "${output_dir}/quality-samples.jsonl" \
  --semantic_evaluator nomic_proxy \
  --overwrite

printf 'Acceptance artifacts: %s\n' "${output_dir}"
