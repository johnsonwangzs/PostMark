# PostMark-Local

PostMark-Local is a fully local reimplementation of the PostMark watermark
pipeline. The implementation and acceptance contract is
`PostMark_local_reimplementation_plan.md`.

## Reproduction scope

The `compat` selector preserves the official two-stage selection rule: keyed
anchor top-`3k` preselection followed by candidate-word embedding reranking to
`k`. The unavailable official anchor embeddings are replaced by a deterministic
Nomic table built from a fingerprinted local C4 shard.

This is not an exact reproduction of the paper's reported numbers. The anchors
are regenerated, the local Llama 3.1 8B inserter is not the inserter used in the
paper's open setup, and the currently implemented detectors are portable
`exact_lemma` and experimental `nomic_fuzzy` variants. Paragram compatibility is
deferred until trusted local raw vectors are provisioned. Portable and Paragram
results must never be combined in one aggregate.

Blind detection here means that the detector does not need `text1` or saved
`list1/list2`. It still requires the same keyed table and selection configuration,
and recomputes expected words from each candidate. This is keyed blind detection,
not keyless detection.

The removed paraphrase branch is not replaced by another attack. Current results
are clean-condition baselines only and make no robustness claim.

## Local resources

The default configuration is `configs/postmark_portable.json` and references:

- Inserter: `/data/llm/Meta-Llama-3.1-8B-Instruct`
- Embedder: `/data/llm/nomic-embed-text-v1`
- Embedder tokenizer: `/data/llm/bert-base-uncased`
- Anchor corpus: `/data/wangzhuoshang/resource/datasets/redpajama-v1-c4/c4-train.00000-of-01024.jsonl`
- spaCy model: `en_core_web_sm`

Paths are configurable. Do not copy models or corpora into the repository.

Generated compatibility resources:

- `resources/candidate_words.json`: 3,266 words in official repository order.
- `resources/postmark_nomic_table.pt`: aligned `(3266, 768)` anchor and candidate
  embeddings built from a deterministic 100K C4 pool. This large file is ignored.
- `resources/postmark_nomic_table.manifest.json`: source, content, model,
  tokenizer, and configuration fingerprints.

## Offline deployment

Model, tokenizer, corpus, spaCy wheel, and Python wheels must be staged before a
formal run. An offline wheelhouse installation can use:

```bash
pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt
python -m spacy validate
```

Every runtime CLI requires both Hugging Face offline controls:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
```

For acceptance testing, `POSTMARK_BLOCK_NETWORK=1` additionally blocks DNS and
outbound socket paths inside the process. It is useful where an OS network
namespace is unavailable; infrastructure-level isolation is still preferred for
formal experiments.

All `from_pretrained()` calls use `local_files_only=True`. Missing local resources
fail instead of falling back to a model hub.

## Build resources

```bash
python -m postmark.build_candidate_words \
  --implementation_profile compat \
  --legacy_pickle_path valid_wtmk_words_in_wiki_base-only-f1000.pkl \
  --output_path resources/candidate_words.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
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

The legacy candidate-word pickle is read through a restricted unpickler and is
converted to JSON. Runtime watermarking and detection do not load it.

## Watermark

Formal baseline comparisons should first generate and freeze one shared `text1`
JSONL for every method:

```json
{"id":"0","text":"Existing model-generated response."}
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python -m postmark.watermark \
  --input_path data/input.jsonl \
  --output_path runs/postmark/watermarked.jsonl \
  --text_field text
```

Prompt mode is supported as a convenience with `--prompt_field` and
`--base_llm_path`, using a stable per-ID sampling seed. It should be used to
prepare and freeze `text1`, not to let different baselines sample different source
answers.

The output stores `text1/list1/text2/list2`, group and retry diagnostics, terminal
status, input hash, selection/run hashes, and resource fingerprints. Resume is
keyed by ID and hashes; conflicts require a new output path or `--overwrite`.

## Blind detection and evaluation

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
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

The threshold uses calibration negatives only and the decision rule
`score >= tau`. Paired output reports held-out ROC-AUC, actual FPR/TPR, means,
split hashes, and deterministic paired-ID percentile bootstrap confidence
intervals. Reports are labeled diagnostic unless calibration and held-out
negative sets each contain at least 1,000 samples.

`nomic_fuzzy` is an experimental portable alternative. Its threshold must be
fixed on development/calibration data and cannot be presented as the paper's
Paragram detector.

## Quality report

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python -m postmark.quality \
  --input_path runs/postmark/watermarked.jsonl \
  --output_path runs/postmark/quality.json \
  --sample_output_path runs/postmark/quality-samples.jsonl \
  --semantic_evaluator nomic_proxy
```

All eligible IDs remain in the denominator, including terminal failures. The
report covers failure classes, insertion success, exhausted attempts, empty
outputs, separate embedding/generation truncation, requested-word presence,
list overlap, absolute/relative length changes, and Nomic-proxy similarity.
Precomputed task scores can be supplied as paired fields with an evaluator
fingerprint. Aggregation rejects mixed run/selection hashes and non-finite values.

## Acceptance and tests

The three-sample real-model acceptance pipeline enables the process network guard:

```bash
CUDA_VISIBLE_DEVICES=1 scripts/run_offline_acceptance.sh
```

Its tiny two-negative calibration fixture is intentionally insufficient for a
formal 1% FPR claim, so the output must say `diagnostic_insufficient_negatives`.

```bash
python -m unittest discover -s tests -v
python -m postmark.watermark --help
python -m postmark.detect --help
python -m postmark.quality --help
```

## Reporting rules and limitations

- Label runs with inserter, selector, table seed/hash, detector, and all config
  hashes. `exact_paper_reproduction` is always false.
- Use the same frozen IDs and `text1` for all baselines. Do not filter by insertion
  success, truncation, list overlap, or output quality after observing results.
- Keep anchor/candidate corpora, development/calibration data, and held-out data
  isolated. Prefer multiple preregistered table seeds.
- Nomic semantic similarity is a proxy, not an independent task-quality judge.
- The local Nomic snapshot's custom loader currently emits PyTorch's trusted-local
  `weights_only=False` future warning. The snapshot must be treated as trusted.
- No Paragram compatibility resource builder/runtime is included in this portable
  milestone. Do not label portable detection as paper-compatible.
