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
paper's open setup, and the detectors are portable `exact_lemma` and
experimental `nomic_fuzzy` variants. Paragram is intentionally out of scope and
will not be implemented. The primary baseline is named
`PostMark-Local-Nomic-Fuzzy`; exact lemma is reported only as an ablation.

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

## Frozen 200-pair experiment data

The formal local baseline uses the ELI5 Llama traces below. The response is the
clean text, and the existing trace length is the token count used for the
`256..512` filter. Seed `1618` fixes all split membership by stable ID hash.

```bash
python -m postmark.prepare_experiment \
  --test_source_path /data/wangzhuoshang/experiment/post_dlm/data/llm_traces_eli5_less_repeat_run_256_512.jsonl \
  --calibration_source_path /data/wangzhuoshang/experiment/post_dlm/data/llm_traces_eli5_less_repeat_calibration.jsonl \
  --output_dir runs/postmark_200/data \
  --id_field record_id \
  --text_field response \
  --token_trace_field trace \
  --test_count 200 \
  --pilot_count 30 \
  --calibration_count 1000 \
  --detector_dev_count 200 \
  --min_tokens 256 \
  --max_tokens 512 \
  --target_fpr 0.01 \
  --seed 1618
```

This writes `pilot.jsonl`, `test_200.jsonl`, `detector_dev_200.jsonl`,
`calibration_1000.jsonl`, and `dataset_manifest.json`. The 30 pilot records and
the disjoint detector-dev negatives may be used to freeze generation and
Nomic-presence settings. The 200 test records and their outcomes must not be used
for tuning. Formal calibration contains negatives only and freezes the decision
threshold at 1% target FPR after detector settings are fixed.

With 200 held-out negative texts, empirical FPR has `0.5%` resolution. The
pipeline therefore labels the 1% FPR result diagnostic even though the threshold
is independently calibrated on 1,000 negatives; this experiment supports the
fixed 200-pair comparison but not a high-precision population FPR claim.

The generation pilot freezes `configs/postmark_200.json`: ratio `0.06`, v2
group size `20`, minimum group presence `0.5`, one attempt per group,
`max_new_tokens=768`, and seed `1618`. On the 30-record pilot this configuration
completed all records with 96.7% insertion success, 0% output truncation, mean
Nomic-proxy similarity `0.949`, and mean relative word-length change `+29.1%`.
These are parameter-selection diagnostics, not held-out detection results.

The disjoint 200-negative detector-dev split was then used with the 30 pilot
pairs to compare preregistered Nomic similarity thresholds `0.70`, `0.75`, and
`0.80`. Threshold `0.80` is frozen in `configs/postmark_200.json`; it produced
diagnostic ROC-AUC `0.9989`, TPR `1.0`, and pilot FPR `0.0333`. These small-split
figures selected a detector setting and are not final reported performance.
The subsequent 1,000-negative formal calibration freezes decision threshold
`tau=0.2380952381` at empirical FPR `0.01`. Dataset, model, selector, detector,
and calibration hashes are bound in `configs/postmark_200_protocol.json`.

The frozen 200-pair run completed with 94.5% insertion success, 1% generation
output truncation, mean Nomic-proxy similarity `0.959`, and mean relative length
change `+33.6%`. Primary Nomic-fuzzy detection reports ROC-AUC `0.99966`, TPR
`0.99`, and held-out FPR `0.01` at the frozen threshold. Exact lemma is an
ablation with ROC-AUC `0.99973`, TPR `0.995`, and FPR `0.015`. Because the held-out
negative count is 200, both metric reports remain explicitly diagnostic for a
population-level 1% FPR claim.

The preserved baseline can be compared with an isolated ratio `0.12`, v2 group
size `10` variant. All other settings and datasets remain fixed. Run the full
resumable generation, quality, recalibration, Nomic detection, and exact-lemma
ablation with:

```bash
CUDA_VISIBLE_DEVICES=1 scripts/run_postmark_200_r012_g10.sh
```

Outputs are written only under `runs/postmark_200_r012_g10/`. The script
recomputes the 1% FPR calibration threshold because changing ratio changes the
selection contract and score discretization, then prints the new threshold
beside the preserved baseline threshold.

The completed variant required recalibration: Nomic decision threshold changed
from `0.238095` to `0.179487` (calibration FPR `0.009`). With the recalibrated
threshold, Nomic detection reached ROC-AUC `1.0`, TPR `1.0`, and held-out FPR
`0.01`. Reusing the baseline threshold would instead give held-out TPR `0.99`
and FPR `0.0` on this sample.

The higher-density variant has a substantial quality cost relative to the
preserved ratio `0.06`/group `20` run: output truncation rises from `1%` to
`26%`, mean relative length growth from `33.6%` to `82.2%`, and mean Nomic-proxy
similarity falls from `0.959` to `0.929`. Its insertion success is `93.5%`
versus `94.5%`, while its mean Nomic detection score gap is lower (`0.3323`
versus `0.4068`).

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

`nomic_fuzzy` is the primary portable detector. Its presence similarity setting
must be frozen using pilot/calibration data and cannot be presented as the
paper's original detector.

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
- Paragram is not part of the project scope. Do not label portable detection as
  an exact detector reproduction or combine its numbers with the paper's table.
