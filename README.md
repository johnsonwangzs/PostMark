# PostMark-Local

This repository is being converted into a fully local reimplementation of the
PostMark watermark baseline. The implementation contract and acceptance criteria
are defined in `PostMark_local_reimplementation_plan.md`.

The current development stage provides deterministic configuration, JSONL,
resource-manifest, and offline CLI foundations. Selector resource construction,
watermark insertion, and portable blind detection are added in subsequent stages;
the CLI entry points intentionally fail with an actionable message until their
required local resource layer is implemented.

## Local resources

The default development configuration is `configs/postmark_portable.json`. It
references the locally provisioned Llama 3.1 inserter, Nomic embedder, BERT
tokenizer, C4 anchor corpus, and `en_core_web_sm`. Paths remain configurable and
large resources must not be copied into this repository.

Runtime execution is offline-only:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The first detector profile is portable `exact_lemma`. The official fuzzy detector
is deferred until its local vector resource is provisioned, and portable results
must be reported separately from paper-compatible results.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m postmark.watermark --help
python -m postmark.detect --help
```
