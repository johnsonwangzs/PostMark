import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DependencyCleanupTests(unittest.TestCase):
    def test_runtime_sources_have_no_hosted_clients_or_attack_pipeline(self):
        forbidden = (
            "import openai",
            "from openai",
            "import together",
            "from together",
            "nltk.download",
            "text3",
            "list3",
        )
        for path in (REPOSITORY_ROOT / "postmark").glob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            for marker in forbidden:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, source)

    def test_requirements_are_small_and_do_not_install_cuda_runtime(self):
        requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).lower()
        forbidden = ("openai", "together", "nvidia-", "torchaudio", "torchvision")
        for marker in forbidden:
            self.assertNotIn(marker, requirements)
        self.assertLess(len(requirements.splitlines()), 20)

    def test_cli_help_is_local_and_has_no_removed_options(self):
        for module in (
            "postmark.watermark",
            "postmark.detect",
            "postmark.build_candidate_words",
            "postmark.build_nomic_anchor_pool",
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=REPOSITORY_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                help_text = result.stdout.lower()
                for marker in ("gpt", "openai", "together", "paraphraser", "--para"):
                    self.assertNotIn(marker, help_text)


if __name__ == "__main__":
    unittest.main()
