import os
import subprocess
import sys
import unittest


class OfflineFoundationTests(unittest.TestCase):
    def test_foundation_imports_do_not_load_models_or_open_sockets(self):
        script = r'''
import socket
import sys

def blocked(*args, **kwargs):
    raise AssertionError("network access attempted during import")

socket.socket = blocked
import postmark.common
import postmark.config
import postmark.detect
import postmark.resources
import postmark.watermark

for forbidden in ("torch", "spacy", "transformers", "openai", "together"):
    assert forbidden not in sys.modules, forbidden
'''
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
