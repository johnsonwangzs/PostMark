from __future__ import annotations

from types import SimpleNamespace


class FakeSelector:
    selection_config_sha256 = "selector-config-fixture"
    selection_config = {"fixture": True}
    table_manifest = SimpleNamespace(content_sha256="selector-resource-fixture")
    config_consistent = True
    eligible_for_aggregate = True
    paper_method_compatible = True
    exact_paper_reproduction = False

    def __init__(self, words: list[str] | None = None) -> None:
        self.words = words or ["alpha", "beta"]

    def word_count_to_k(self, text: str) -> int:
        return len(self.words)

    def select_words(self, text: str, *, top_k: int | None = None) -> list[str]:
        k = len(self.words) if top_k is None else top_k
        return self.words[:k]


class QueueInserter:
    fingerprint = {"fixture": "queue-inserter"}

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EchoInserter:
    fingerprint = {"fixture": "echo-inserter"}

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        text, words = prompt.split("\nWORDS=", 1)
        return text.removeprefix("TEXT=") + " " + words.replace(",", " ")


class SamplingBase:
    fingerprint = {"fixture": "sampling-base"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs["seed"]))
        return f"base response for {prompt} seed {kwargs['seed']}"
