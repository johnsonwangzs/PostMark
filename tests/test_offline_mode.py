from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from postmark.common import ConfigurationError, require_offline_environment


class OfflineRuntimeTests(unittest.TestCase):
    def test_both_huggingface_offline_variables_are_required(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            with self.assertRaisesRegex(ConfigurationError, "HF_HUB_OFFLINE"):
                require_offline_environment()
            os.environ["HF_HUB_OFFLINE"] = "1"
            with self.assertRaisesRegex(ConfigurationError, "TRANSFORMERS_OFFLINE"):
                require_offline_environment()
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            require_offline_environment()

    def test_network_guard_blocks_dns_and_outbound_socket_paths(self) -> None:
        script = r'''
import os
import socket
import torch
from postmark.common import NetworkAccessError, install_network_guard_from_environment

os.environ["POSTMARK_BLOCK_NETWORK"] = "1"
assert install_network_guard_from_environment() is True
assert install_network_guard_from_environment() is True

operations = [
    lambda: socket.getaddrinfo("example.com", 443),
    lambda: socket.create_connection(("127.0.0.1", 9)),
    lambda: socket.socket().connect(("127.0.0.1", 9)),
    lambda: socket.socket().connect_ex(("127.0.0.1", 9)),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("127.0.0.1", 9)),
]
for operation in operations:
    try:
        operation()
    except NetworkAccessError:
        pass
    else:
        raise AssertionError("network operation was not blocked")
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

    def test_runtime_cli_fails_before_loading_resources_without_offline_env(self) -> None:
        environment = os.environ.copy()
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "postmark.watermark",
                "--input_path",
                "does-not-exist.jsonl",
                "--output_path",
                "unused.jsonl",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Offline runtime requires", result.stderr)
        self.assertNotIn("Loading checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
