# PostMark-Local

This repository is being converted into a fully local reimplementation of the
PostMark watermark baseline. The implementation contract and acceptance criteria
are defined in `PostMark_local_reimplementation_plan.md`.

The current development stage provides deterministic configuration and JSONL,
resource manifests, canonical candidate-word conversion, a reproducible local
Nomic anchor table, the official two-stage selector, and local Llama watermark
insertion with ID-based resume. Portable blind detection, independent calibration,
and paired bootstrap metrics are also available. Quality and failure-rate
aggregation is the next stage.

## Local resources

The default development configuration is `configs/postmark_portable.json`. It
references the locally provisioned Llama 3.1 inserter, Nomic embedder, BERT
tokenizer, C4 anchor corpus, and `en_core_web_sm`. Paths remain configurable and
large resources must not be copied into this repository.

The generated compatibility resources are:

- `resources/candidate_words.json`: 3,266 words, preserving official repository order.
- `resources/postmark_nomic_table.pt`: 100K C4 pool mapped to aligned `(3266, 768)`
  anchor and candidate-word embeddings.
- `resources/postmark_nomic_table.manifest.json`: canonical content and source
  fingerprints. The `.pt` artifact is local-only and ignored by Git.

Build them with:

```bash
python -m postmark.build_candidate_words \
  --implementation_profile compat \
  --legacy_pickle_path valid_wtmk_words_in_wiki_base-only-f1000.pkl \
  --output_path resources/candidate_words.json

python -m postmark.build_nomic_anchor_pool \
  --implementation_profile compat \
  --selection_mode official_two_stage \
  --input_path /path/to/c4-train.00000-of-01024.jsonl \
  --candidate_words_path resources/candidate_words.json \
  --embedder_path /path/to/nomic-embed-text-v1 \
  --tokenizer_path /path/to/bert-base-uncased \
  --output_path resources/postmark_nomic_table.pt \
  --corpus_revision redpajama-data-1T-v1.0.0-c4-00000-of-01024 \
  --num_anchor_chunks 100000 \
  --seed 42 \
  --local_files_only
```

Runtime execution is offline-only:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Insert a watermark into an existing-text JSONL with the default local resources:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m postmark.watermark \
  --input_path data/input.jsonl \
  --output_path runs/postmark/watermarked.jsonl \
  --text_field text
```

Each output record contains `text1`, `list1`, `text2`, `list2`, insertion
diagnostics, and resource/configuration hashes. Resume is keyed by stable sample
ID and refuses input, model, prompt, or configuration conflicts. Use
`--overwrite` only when intentionally starting that output path again.

Portable blind detection supports `exact_lemma` and the explicitly experimental
`nomic_fuzzy` mode. Both recompute expected words from each candidate using the
same keyed selector table; neither reads insertion-time `list1/list2`. The official
Paragram fuzzy detector is deferred until its local vector resource is provisioned,
and portable results must be reported separately from paper-compatible results.

Run portable paired detection with an independent negative-only calibration set:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m postmark.detect \
  --input_path runs/postmark/watermarked.jsonl \
  --output_path runs/postmark/scored-exact-lemma.jsonl \
  --paired \
  --negative_field text1 \
  --positive_field text2 \
  --calibration_path data/calibration.jsonl \
  --calibration_text_field text \
  --presence_mode exact_lemma
```

The default 1% FPR report is labeled diagnostic unless both calibration and
held-out negative sets contain at least 1,000 samples. Nomic-fuzzy thresholds must
be fixed on development/calibration data and must not be presented as compatible
with the paper's Paragram detector.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m postmark.watermark --help
python -m postmark.detect --help
```
