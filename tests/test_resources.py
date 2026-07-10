import shutil
import tempfile
import unittest
from pathlib import Path

from postmark.common import (
    ConfigurationError,
    ResourceError,
    ResourceMismatchError,
    sha256_file,
)
from postmark.resources import (
    ResourceManifest,
    build_file_manifest,
    fingerprint_path,
    load_manifest,
    verify_manifest,
    write_manifest,
)


class ResourceManifestTests(unittest.TestCase):
    def test_manifest_round_trip_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = root / "resource.bin"
            manifest_path = root / "resource.manifest.json"
            content.write_bytes(b"postmark-resource")
            manifest = build_file_manifest(
                content,
                resource_type="test_resource",
                metadata={"seed": 42},
            )
            write_manifest(manifest_path, manifest)
            loaded = load_manifest(manifest_path)
            self.assertEqual(loaded, manifest)
            verify_manifest(
                loaded,
                computed_content_sha256=sha256_file(content),
                expected_resource_type="test_resource",
                expected_resource_version=1,
            )

    def test_tampered_content_is_rejected(self):
        manifest = ResourceManifest(
            resource_type="test",
            resource_version=1,
            content_sha256="0" * 64,
        )
        with self.assertRaises(ResourceMismatchError):
            verify_manifest(manifest, computed_content_sha256="1" * 64)

    def test_malformed_manifest_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            ResourceManifest(
                resource_type="test",
                resource_version=True,
                content_sha256="0" * 64,
            )

    def test_directory_fingerprint_is_path_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            (first / "nested").mkdir(parents=True)
            (first / "config.json").write_text("{}\n", encoding="utf-8")
            (first / "nested" / "weights.bin").write_bytes(b"weights")
            shutil.copytree(first, second)
            self.assertEqual(fingerprint_path(first), fingerprint_path(second))

    def test_directory_fingerprint_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"before")
            before = fingerprint_path(root)
            (root / "weights.bin").write_bytes(b"after")
            self.assertNotEqual(before.sha256, fingerprint_path(root).sha256)

    def test_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("data", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(ResourceError):
                fingerprint_path(link)


if __name__ == "__main__":
    unittest.main()
