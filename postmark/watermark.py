"""Offline PostMark watermark insertion and resumable JSONL pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .common import (
    ConfigurationError,
    DuplicateIdError,
    GenerationError,
    JsonlError,
    SelectionError,
    append_jsonl_record,
    atomic_write_json,
    atomic_write_jsonl,
    derive_sample_seed,
    load_json_object,
    load_jsonl,
    recover_truncated_jsonl_tail,
    sha256_file,
    sha256_json,
    stable_content_id,
    stable_word_count,
)
from .config import PostMarkConfig


TERMINAL_STATUSES = {"completed", "failed"}
RUN_MANIFEST_VERSION = 1


class Generator(Protocol):
    fingerprint: Mapping[str, Any]

    def generate(self, prompt: str, **kwargs: Any) -> Any: ...


class Selector(Protocol):
    selection_config_sha256: str
    selection_config: Mapping[str, Any]
    table_manifest: Any
    config_consistent: bool
    eligible_for_aggregate: bool
    paper_method_compatible: bool
    exact_paper_reproduction: bool

    def word_count_to_k(self, text: str) -> int: ...

    def select_words(self, text: str, *, top_k: int | None = None) -> list[str]: ...


def _component_fingerprint(component: Any) -> Mapping[str, Any]:
    fingerprint = getattr(component, "fingerprint", None)
    if not isinstance(fingerprint, Mapping):
        raise ConfigurationError("Local generator must expose a mapping fingerprint")
    return fingerprint


def _contains(text: str, word: str) -> bool:
    return word.lower() in text.lower()


def _presence(text: str, words: Sequence[str]) -> float:
    if not words:
        return 0.0
    return sum(_contains(text, word) for word in words) / len(words)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _generation_text(result: Any) -> tuple[str, bool, bool]:
    if isinstance(result, str):
        return result.strip(), False, False
    if (
        isinstance(getattr(result, "text", None), str)
        and isinstance(getattr(result, "input_truncated", None), bool)
        and isinstance(getattr(result, "output_truncated", None), bool)
    ):
        return result.text.strip(), result.input_truncated, result.output_truncated
    raise GenerationError("Generator returned an unsupported result type")


class PostMarkWatermarker:
    def __init__(
        self,
        inserter: Generator,
        selector: Selector,
        *,
        prompt_path: str,
        iterate: str = "v2",
        group_size: int = 10,
        min_group_presence: float = 0.5,
        max_insert_attempts: int = 1,
        retry_strategy: str = "missing_words",
        max_new_tokens: int = 600,
        min_watermark_words: int | None = None,
        max_watermark_words: int | None = None,
        seed: int = 42,
    ) -> None:
        if iterate != "v2":
            raise ConfigurationError("Only iterative insertion v2 is supported")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (group_size, max_insert_attempts, max_new_tokens)
        ):
            raise ConfigurationError("Group size, attempts, and max_new_tokens must be positive")
        if (
            isinstance(min_group_presence, bool)
            or not isinstance(min_group_presence, (int, float))
            or not 0 <= min_group_presence <= 1
        ):
            raise ConfigurationError("min_group_presence must be in [0, 1]")
        if retry_strategy != "missing_words":
            raise ConfigurationError("Only missing_words retry is supported")
        for name, value in (
            ("min_watermark_words", min_watermark_words),
            ("max_watermark_words", max_watermark_words),
        ):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ConfigurationError(f"{name} must be a non-negative integer")
        if (
            min_watermark_words is not None
            and max_watermark_words is not None
            and min_watermark_words > max_watermark_words
        ):
            raise ConfigurationError("min_watermark_words exceeds max_watermark_words")

        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.is_file():
            raise ConfigurationError(f"Insertion prompt does not exist: {self.prompt_path}")
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
        try:
            self.prompt_template.format("text", "words")
        except (IndexError, KeyError, ValueError) as exc:
            raise ConfigurationError("Insertion prompt must contain two positional fields") from exc

        self.inserter = inserter
        self.selector = selector
        self.iterate = iterate
        self.group_size = group_size
        self.min_group_presence = float(min_group_presence)
        self.max_insert_attempts = max_insert_attempts
        self.retry_strategy = retry_strategy
        self.max_new_tokens = max_new_tokens
        self.min_watermark_words = min_watermark_words
        self.max_watermark_words = max_watermark_words
        self.seed = seed
        self.selection_config = {
            "selector_selection_config_sha256": selector.selection_config_sha256,
            "selector_selection_config": dict(selector.selection_config),
            "min_watermark_words": min_watermark_words,
            "max_watermark_words": max_watermark_words,
        }
        self.selection_config_sha256 = sha256_json(self.selection_config)
        self.run_config = {
            "version": RUN_MANIFEST_VERSION,
            "selection_config_sha256": self.selection_config_sha256,
            "prompt_sha256": sha256_file(self.prompt_path),
            "inserter_fingerprint": dict(_component_fingerprint(inserter)),
            "decoding": {
                "temperature": 0.0,
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
            },
            "iterate": iterate,
            "group_size": group_size,
            "min_group_presence": self.min_group_presence,
            "max_insert_attempts": max_insert_attempts,
            "retry_strategy": retry_strategy,
            "insertion_presence_mode": "compat_substring_case_insensitive",
            "seed": seed,
        }
        self.run_config_sha256 = sha256_json(self.run_config)

    @property
    def selector_resource_sha256(self) -> str:
        value = getattr(self.selector.table_manifest, "content_sha256", None)
        if not isinstance(value, str) or not value:
            raise ConfigurationError("Selector has no resource content fingerprint")
        return value

    def _select_words(self, text: str) -> list[str]:
        k = self.selector.word_count_to_k(text)
        if self.min_watermark_words is not None:
            k = max(k, self.min_watermark_words)
        if self.max_watermark_words is not None:
            k = min(k, self.max_watermark_words)
        return self.selector.select_words(text, top_k=k)

    def insert_watermark(self, text: str, *, sample_id: str = "sample") -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ConfigurationError("Watermark input text must be non-empty")
        try:
            list1 = self._select_words(text)
        except SelectionError as exc:
            reason = (
                "k_exceeds_vocabulary"
                if "exceeds candidate vocabulary" in str(exc)
                else "selection_failed"
            )
            return self._failed_result(text, [], reason, str(exc))
        if not list1:
            return self._failed_result(text, [], "k_zero", "No watermark words selected")

        current_text = text
        attempts: list[dict[str, Any]] = []
        attempted_generation = 0
        usable_generation = 0
        any_input_truncated = False
        any_output_truncated = False
        groups = [list1[start : start + self.group_size] for start in range(0, len(list1), self.group_size)]

        for group_index, group in enumerate(groups):
            best_text = current_text
            best_presence = _presence(best_text, group)
            seen_prompts: set[str] = set()
            group_stop_reason = "presence_threshold_met"
            if best_presence < self.min_group_presence:
                group_stop_reason = "attempt_limit"
                for attempt_index in range(self.max_insert_attempts):
                    requested = [word for word in group if not _contains(best_text, word)]
                    if not requested:
                        group_stop_reason = "all_words_present"
                        break
                    prompt = self.prompt_template.format(best_text, ", ".join(requested))
                    prompt_sha256 = sha256_json({"prompt": prompt})
                    if prompt_sha256 in seen_prompts:
                        group_stop_reason = "duplicate_prompt"
                        break
                    seen_prompts.add(prompt_sha256)
                    attempted_generation += 1
                    diagnostic: dict[str, Any] = {
                        "group_index": group_index,
                        "attempt_index": attempt_index,
                        "prompt_sha256": prompt_sha256,
                        "requested_words": requested,
                        "presence_before": best_presence,
                        "selected": False,
                    }
                    try:
                        generated = self.inserter.generate(
                            prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=0.0,
                            do_sample=False,
                        )
                        candidate, input_truncated, output_truncated = _generation_text(generated)
                        diagnostic["input_truncated"] = input_truncated
                        diagnostic["output_truncated"] = output_truncated
                        any_input_truncated |= input_truncated
                        any_output_truncated |= output_truncated
                    except (GenerationError, RuntimeError, ValueError, TypeError) as exc:
                        candidate = ""
                        diagnostic["error"] = str(exc)
                    if not candidate:
                        diagnostic.update(
                            {
                                "presence": best_presence,
                                "missing_words": requested,
                                "stop_reason": "empty_or_error",
                            }
                        )
                        attempts.append(diagnostic)
                        group_stop_reason = "empty_or_error"
                        continue

                    usable_generation += 1
                    candidate_presence = _presence(candidate, group)
                    missing_after = [word for word in group if not _contains(candidate, word)]
                    improved = candidate_presence > best_presence
                    diagnostic.update(
                        {
                            "presence": candidate_presence,
                            "missing_words": missing_after,
                            "selected": improved,
                        }
                    )
                    if improved:
                        best_text = candidate
                        best_presence = candidate_presence
                    if best_presence >= self.min_group_presence:
                        diagnostic["stop_reason"] = "presence_threshold_met"
                        group_stop_reason = "presence_threshold_met"
                        attempts.append(diagnostic)
                        break
                    if not improved:
                        diagnostic["stop_reason"] = "no_improvement"
                        group_stop_reason = "no_improvement"
                        attempts.append(diagnostic)
                        break
                    diagnostic["stop_reason"] = "retry_missing_words"
                    attempts.append(diagnostic)
                if attempts and attempts[-1].get("group_index") == group_index:
                    attempts[-1]["group_stop_reason"] = group_stop_reason
            current_text = best_text

        if attempted_generation and not usable_generation:
            return self._failed_result(
                text,
                list1,
                "generation_failed",
                "All insertion generations were empty or failed",
                attempts=attempts,
            )

        text2 = current_text
        try:
            list2 = self._select_words(text2)
            list2_failure = None
        except SelectionError as exc:
            list2 = []
            list2_failure = str(exc)
        requested_presence = _presence(text2, list1)
        diagnostics = {
            "list_overlap": _jaccard(list1, list2),
            "requested_word_presence": requested_presence,
            "num_groups": len(groups),
            "num_attempts": attempted_generation,
            "attempts": attempts,
            "insertion_failed": requested_presence < self.min_group_presence or bool(list2_failure),
            "embedding_input_truncated": False,
            "generation_input_truncated": any_input_truncated,
            "generation_output_truncated": any_output_truncated,
            "length_delta_words": stable_word_count(text2) - stable_word_count(text),
            "stop_reason": "all_groups_processed",
        }
        if list2_failure:
            diagnostics["list2_failure"] = list2_failure
        return {
            "status": "completed",
            "text1": text,
            "list1": list1,
            "text2": text2,
            "list2": list2,
            "diagnostics": diagnostics,
        }

    def _failed_result(
        self,
        text: str,
        list1: list[str],
        reason: str,
        detail: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "text1": text,
            "list1": list1,
            "text2": text,
            "list2": list1.copy(),
            "diagnostics": {
                "list_overlap": _jaccard(list1, list1),
                "requested_word_presence": _presence(text, list1),
                "num_groups": (len(list1) + self.group_size - 1) // self.group_size,
                "num_attempts": len(attempts or []),
                "attempts": attempts or [],
                "insertion_failed": True,
                "embedding_input_truncated": False,
                "generation_input_truncated": False,
                "generation_output_truncated": False,
                "length_delta_words": 0,
                "stop_reason": reason,
                "failure_reason": reason,
                "failure_detail": detail,
            },
        }


def _sample_id(record: Mapping[str, Any], id_field: str | None) -> str:
    if id_field and id_field in record:
        value = record[id_field]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
            raise JsonlError(f"Sample ID field {id_field!r} must be a non-empty string or integer")
        return str(value)
    return stable_content_id(record)


def _manifest_path(output_path: Path, explicit: str | None) -> Path:
    return Path(explicit) if explicit else output_path.with_name(output_path.name + ".manifest.json")


def run_watermark_pipeline(
    *,
    input_path: str,
    output_path: str,
    watermarker: PostMarkWatermarker,
    text_field: str | None = None,
    prompt_field: str | None = None,
    id_field: str | None = "id",
    base_llm: Generator | None = None,
    base_max_new_tokens: int = 600,
    seed: int = 42,
    run_manifest_path: str | None = None,
    retry_failed: bool = False,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    text_mode = bool(text_field)
    prompt_mode = bool(prompt_field)
    if text_mode == prompt_mode:
        raise ConfigurationError("Choose exactly one of text_field or prompt_field")
    if prompt_mode and base_llm is None:
        raise ConfigurationError("prompt_field requires a local base_llm")
    if retry_failed and not overwrite and Path(output_path).exists():
        raise ConfigurationError("retry_failed requires a new output path or --overwrite")

    inputs = load_jsonl(input_path, limit=limit)
    indexed_inputs: list[tuple[str, dict[str, Any], str]] = []
    seen_input_ids: set[str] = set()
    for record in inputs:
        sample_id = _sample_id(record, id_field)
        if sample_id in seen_input_ids:
            raise DuplicateIdError(f"Duplicate input sample ID {sample_id!r}")
        seen_input_ids.add(sample_id)
        indexed_inputs.append((sample_id, record, sha256_json(record)))

    base_fingerprint = dict(_component_fingerprint(base_llm)) if base_llm is not None else None
    pipeline_config = {
        **watermarker.run_config,
        "base_llm_fingerprint": base_fingerprint,
        "base_decoding": (
            {"temperature": 1.0, "top_p": 0.9, "do_sample": True, "max_new_tokens": base_max_new_tokens}
            if base_llm is not None
            else None
        ),
        "input_mode": "text" if text_mode else "prompt",
        "text_field": text_field,
        "prompt_field": prompt_field,
        "id_field": id_field,
        "retry_failed": retry_failed,
        "seed": seed,
    }
    run_config_sha256 = sha256_json(pipeline_config)
    output = Path(output_path)
    manifest_path = _manifest_path(output, run_manifest_path)
    if overwrite:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    if output.exists() and not manifest_path.exists():
        raise ConfigurationError(
            "Existing output has no run manifest; use --overwrite or a new output path"
        )
    if manifest_path.exists():
        manifest_records = load_json_object(manifest_path)
        if manifest_records.get("run_config_sha256") != run_config_sha256:
            raise ConfigurationError("Existing run manifest differs; use --overwrite or a new output path")
    else:
        atomic_write_json(
            manifest_path,
            {
                "version": RUN_MANIFEST_VERSION,
                "run_config_sha256": run_config_sha256,
                "selection_config_sha256": watermarker.selection_config_sha256,
                "selector_resource_sha256": watermarker.selector_resource_sha256,
                "config": pipeline_config,
                "config_consistent": watermarker.selector.config_consistent,
                "eligible_for_aggregate": watermarker.selector.eligible_for_aggregate,
                "paper_method_compatible": watermarker.selector.paper_method_compatible,
                "exact_paper_reproduction": watermarker.selector.exact_paper_reproduction,
            },
        )

    existing: dict[str, dict[str, Any]] = {}
    nonterminal_ids: set[str] = set()
    if output.exists():
        recover_truncated_jsonl_tail(output)
        output_records = load_jsonl(output)
        for record in output_records:
            sample_id = _sample_id(record, "id")
            if sample_id in existing:
                raise DuplicateIdError(f"Duplicate output sample ID {sample_id!r}")
            existing[sample_id] = record
            if record.get("status") not in TERMINAL_STATUSES:
                nonterminal_ids.add(sample_id)
        if nonterminal_ids:
            input_contracts = {
                sample_id: input_sha256
                for sample_id, _, input_sha256 in indexed_inputs
            }
            for sample_id in nonterminal_ids:
                if sample_id not in input_contracts:
                    raise ConfigurationError(
                        f"Cannot rerun nonterminal id {sample_id!r}: it is absent from input"
                    )
                record = existing[sample_id]
                expected = {
                    "input_sha256": input_contracts[sample_id],
                    "selection_config_sha256": watermarker.selection_config_sha256,
                    "run_config_sha256": run_config_sha256,
                }
                conflicts = [
                    key for key, value in expected.items() if record.get(key) != value
                ]
                if conflicts:
                    raise ConfigurationError(
                        f"Resume conflict for nonterminal id {sample_id!r}: "
                        f"{', '.join(conflicts)}"
                    )
            atomic_write_jsonl(
                output,
                (record for sample_id, record in existing.items() if sample_id not in nonterminal_ids),
            )
            for sample_id in nonterminal_ids:
                del existing[sample_id]

    written = 0
    skipped = 0
    for sample_id, record, input_sha256 in indexed_inputs:
        previous = existing.get(sample_id)
        if previous is not None:
            expected = {
                "input_sha256": input_sha256,
                "selection_config_sha256": watermarker.selection_config_sha256,
                "run_config_sha256": run_config_sha256,
            }
            conflicts = [key for key, value in expected.items() if previous.get(key) != value]
            if conflicts:
                raise ConfigurationError(
                    f"Resume conflict for id {sample_id!r}: {', '.join(conflicts)}; "
                    "use --overwrite or a new output path"
                )
            if previous.get("status") not in TERMINAL_STATUSES:
                raise JsonlError(f"Unexpected nonterminal output for id {sample_id!r}")
            skipped += 1
            continue

        source_field = text_field if text_mode else prompt_field
        source = record.get(source_field) if source_field else None
        if not isinstance(source, str) or not source.strip():
            raise JsonlError(f"Input id {sample_id!r} has no non-empty {source_field!r} string")
        if text_mode:
            text1 = source
        else:
            generated = base_llm.generate(  # type: ignore[union-attr]
                source,
                max_new_tokens=base_max_new_tokens,
                temperature=1.0,
                top_p=0.9,
                do_sample=True,
                seed=derive_sample_seed(seed, sample_id),
            )
            text1, _, _ = _generation_text(generated)
            if not text1:
                raise GenerationError(f"Base LLM returned empty text for id {sample_id!r}")
        result = watermarker.insert_watermark(text1, sample_id=sample_id)
        output_record = {
            "id": sample_id,
            **result,
            "input_sha256": input_sha256,
            "selection_config_sha256": watermarker.selection_config_sha256,
            "run_config_sha256": run_config_sha256,
            "selector_resource_sha256": watermarker.selector_resource_sha256,
            "config_consistent": watermarker.selector.config_consistent,
            "eligible_for_aggregate": watermarker.selector.eligible_for_aggregate,
            "paper_method_compatible": watermarker.selector.paper_method_compatible,
            "exact_paper_reproduction": watermarker.selector.exact_paper_reproduction,
        }
        append_jsonl_record(output, output_record)
        existing[sample_id] = output_record
        written += 1
    return {"input": len(indexed_inputs), "written": written, "skipped": skipped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Insert PostMark-Local watermarks using local resources.")
    parser.add_argument("--config", default="configs/postmark_portable.json")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_field")
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
    parser.add_argument("--implementation_profile", choices=("compat", "portable"))
    parser.add_argument("--selection_mode", choices=("official_two_stage", "anchor_only", "direct_word"))
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--min_watermark_words", type=int)
    parser.add_argument("--max_watermark_words", type=int)
    parser.add_argument("--iterate", choices=("v2",))
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--min_group_presence", type=float)
    parser.add_argument("--max_insert_attempts", type=int)
    parser.add_argument("--retry_strategy", choices=("missing_words",))
    parser.add_argument("--insertion_presence_mode", choices=("compat_substring_case_insensitive",), default="compat_substring_case_insensitive")
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--torch_dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_manifest_path")
    parser.add_argument("--allow_resource_mismatch", action="store_true")
    parser.add_argument("--allow_config_mismatch", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .hf_llm import LocalHFLLM
    from .nomic_embedder import NomicPostMarkEmbedder

    if not args.local_files_only:
        raise ConfigurationError("PostMark-Local requires --local_files_only")
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = PostMarkConfig.load(config_path)
    paths = config.paths.resolved(project_root)
    text_field = args.text_field
    prompt_field = args.prompt_field
    if text_field is None and prompt_field is None:
        text_field = "text"
    if text_field and prompt_field:
        raise ConfigurationError("--text_field and --prompt_field are mutually exclusive")
    if prompt_field and not args.base_llm_path:
        raise ConfigurationError("--prompt_field requires --base_llm_path")

    inserter_path = args.inserter_path or str(paths["inserter"])
    inserter_tokenizer = args.inserter_tokenizer_path or inserter_path
    llm_kwargs = {
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "use_chat_template": args.use_chat_template,
        "local_files_only": True,
    }
    inserter = LocalHFLLM(inserter_path, tokenizer_path=inserter_tokenizer, **llm_kwargs)
    base_llm = None
    if args.base_llm_path:
        base_tokenizer = args.base_tokenizer_path or args.base_llm_path
        if args.base_llm_path == inserter_path and base_tokenizer == inserter_tokenizer:
            base_llm = inserter
        else:
            base_llm = LocalHFLLM(args.base_llm_path, tokenizer_path=base_tokenizer, **llm_kwargs)

    selector = NomicPostMarkEmbedder(
        args.embedder_path or str(paths["embedder"]),
        args.table_path or str(paths["nomic_table"]),
        tokenizer_path=args.embedder_tokenizer_path or str(paths["embedder_tokenizer"]),
        implementation_profile=args.implementation_profile or config.implementation_profile,
        selection_mode=args.selection_mode or config.selection_mode,
        ratio=args.ratio if args.ratio is not None else config.selection.ratio,
        max_length=config.embedding.max_length,
        local_files_only=True,
        allow_resource_mismatch=args.allow_resource_mismatch,
        allow_config_mismatch=args.allow_config_mismatch,
    )
    insertion = config.insertion
    watermarker = PostMarkWatermarker(
        inserter,
        selector,
        prompt_path=args.prompt_path or str(paths["insertion_prompt"]),
        iterate=args.iterate or insertion.iterate,
        group_size=(args.group_size if args.group_size is not None else insertion.group_size),
        min_group_presence=(args.min_group_presence if args.min_group_presence is not None else insertion.min_group_presence),
        max_insert_attempts=(
            args.max_insert_attempts
            if args.max_insert_attempts is not None
            else insertion.max_insert_attempts
        ),
        retry_strategy=args.retry_strategy or "missing_words",
        max_new_tokens=(
            args.max_new_tokens if args.max_new_tokens is not None else insertion.max_new_tokens
        ),
        min_watermark_words=args.min_watermark_words,
        max_watermark_words=args.max_watermark_words,
        seed=args.seed if args.seed is not None else config.runtime.seed,
    )
    result = run_watermark_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        watermarker=watermarker,
        text_field=text_field,
        prompt_field=prompt_field,
        id_field=args.id_field,
        base_llm=base_llm,
        seed=args.seed if args.seed is not None else config.runtime.seed,
        run_manifest_path=args.run_manifest_path,
        retry_failed=args.retry_failed,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"processed={result['input']} written={result['written']} skipped={result['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
