# PostMark-Local

This repository is being converted into a fully local PostMark baseline. Treat
`PostMark_local_reimplementation_plan.md` as the implementation source of truth.

## Local resources

- Inserter: `/data/llm/Meta-Llama-3.1-8B-Instruct`
- Embedder: `/data/llm/nomic-embed-text-v1`
- Embedder tokenizer: `/data/llm/bert-base-uncased`
- Anchor corpus: `/data/wangzhuoshang/resource/datasets/redpajama-v1-c4/c4-train.00000-of-01024.jsonl`
- spaCy model: `en_core_web_sm`

Keep model and dataset paths configurable; do not copy large resources into the
repository. All runtime model loads must use local files with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Implementation rules

- Preserve the official two-stage selector described in the plan.
- Start with the portable detector; Paragram compatibility is deferred.
- Do not add OpenAI, Together.ai, paraphrasing, or other network dependencies.
- Record resource and configuration fingerprints in generated manifests.
- Prefer `CUDA_VISIBLE_DEVICES=1` for GPU validation unless unavailable or overridden.
- Preserve unrelated existing changes and generated experiment outputs.
- Add focused tests for resource loading, selection, detection, resume, and the
  offline JSONL pipeline; run them before considering a phase complete.
