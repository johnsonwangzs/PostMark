#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export POSTMARK_BLOCK_NETWORK=1

CONFIG="configs/postmark_200_r012_g10.json"
INPUT="runs/postmark_200/data/test_200.jsonl"
CALIBRATION="runs/postmark_200/data/calibration_1000.jsonl"
OUTPUT_ROOT="runs/postmark_200_r012_g10/formal"
WATERMARKED="$OUTPUT_ROOT/watermarked.jsonl"
QUALITY="$OUTPUT_ROOT/quality.json"
QUALITY_SAMPLES="$OUTPUT_ROOT/quality-samples.jsonl"
NOMIC_SCORED="$OUTPUT_ROOT/scored-nomic.jsonl"
EXACT_SCORED="$OUTPUT_ROOT/scored-exact-lemma.jsonl"

mkdir -p "$OUTPUT_ROOT"

python -m postmark.watermark \
  --config "$CONFIG" \
  --input_path "$INPUT" \
  --output_path "$WATERMARKED" \
  --text_field text

if [[ ! -e "$QUALITY" && ! -e "$QUALITY_SAMPLES" ]]; then
  python -m postmark.quality \
    --config "$CONFIG" \
    --input_path "$WATERMARKED" \
    --output_path "$QUALITY" \
    --sample_output_path "$QUALITY_SAMPLES" \
    --semantic_evaluator nomic_proxy
else
  echo "Skipping existing quality outputs under $OUTPUT_ROOT"
fi

if [[ ! -e "$NOMIC_SCORED" ]]; then
  python -m postmark.detect \
    --config "$CONFIG" \
    --input_path "$WATERMARKED" \
    --output_path "$NOMIC_SCORED" \
    --paired \
    --negative_field text1 \
    --positive_field text2 \
    --calibration_path "$CALIBRATION" \
    --calibration_text_field text \
    --target_fpr 0.01 \
    --bootstrap_seed 1618
else
  echo "Skipping existing Nomic detection output $NOMIC_SCORED"
fi

if [[ ! -e "$EXACT_SCORED" ]]; then
  python -m postmark.detect \
    --config "$CONFIG" \
    --input_path "$WATERMARKED" \
    --output_path "$EXACT_SCORED" \
    --paired \
    --negative_field text1 \
    --positive_field text2 \
    --calibration_path "$CALIBRATION" \
    --calibration_text_field text \
    --presence_mode exact_lemma \
    --target_fpr 0.01 \
    --bootstrap_seed 1618
else
  echo "Skipping existing exact-lemma output $EXACT_SCORED"
fi

python - "$QUALITY" "$NOMIC_SCORED.manifest.json" \
  "$EXACT_SCORED.manifest.json" "configs/postmark_200_protocol.json" <<'PY'
import json
import sys
from pathlib import Path

quality = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
nomic = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["metrics"]
exact = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["metrics"]
baseline = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
old_tau = baseline["calibration"]["threshold"]
new_tau = nomic["threshold"]

print("\nPostMark r012/g10 completed")
print(f"quality insertion_success={quality['rates']['insertion_success']:.6f}")
print(f"quality output_truncated={quality['rates']['generation_output_truncated']:.6f}")
print(f"quality semantic_mean={quality['distributions']['semantic_similarity']['mean']:.6f}")
print(f"nomic tau={new_tau:.12f} baseline_tau={old_tau:.12f}")
print(f"nomic recalibration_changed_tau={new_tau != old_tau}")
print(
    f"nomic auc={nomic['roc_auc']:.6f} tpr={nomic['heldout_tpr']:.6f} "
    f"fpr={nomic['heldout_fpr']:.6f}"
)
print(
    f"exact auc={exact['roc_auc']:.6f} tpr={exact['heldout_tpr']:.6f} "
    f"fpr={exact['heldout_fpr']:.6f}"
)
PY
