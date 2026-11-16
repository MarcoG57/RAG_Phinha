"""Acquisition integrity with an in-memory public-HTTP boundary and temporary files."""
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_skillcorner as fetcher


class SkillCornerAcquisitionTests(unittest.TestCase):
    def test_verifies_existing_metadata_against_the_pinned_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            path.write_bytes(b"altered")
            with mock.patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(b"pinned")):
                with self.assertRaisesRegex(ValueError, "existing input differs"):
                    fetcher.fetch("https://example.invalid/pinned", path)
            self.assertEqual(path.read_bytes(), b"altered")

    def test_invalid_transfer_never_becomes_an_input_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracking.jsonl"
            expected = (hashlib.sha256(b"valid").hexdigest(), 5)
            with mock.patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    fetcher.fetch("https://example.invalid/tracking", path, expected)
            self.assertFalse(path.exists())
            self.assertEqual(path.with_name(path.name + ".part").read_bytes(), b"wrong")

    def test_destination_appearing_at_publication_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracking.jsonl"
            payload = b"new content"
            expected = (hashlib.sha256(payload).hexdigest(), len(payload))
            original_exists = Path.exists
            checks = 0
            def exists_with_race(candidate):
                nonlocal checks
                if candidate == path:
                    checks += 1
                    if checks == 2:
                        candidate.write_bytes(b"concurrent content")
                        return False
                return original_exists(candidate)
            with mock.patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(payload)), \
                 mock.patch.object(Path, "exists", exists_with_race):
                with self.assertRaises((FileExistsError, ValueError)):
                    fetcher.fetch("https://example.invalid/tracking", path, expected)
            self.assertEqual(path.read_bytes(), b"concurrent content")
            self.assertEqual(path.with_name(path.name + ".part").read_bytes(), payload)

    def test_posix_publication_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracking.jsonl"
            partial = Path(tmp) / "tracking.jsonl.part"
            path.write_bytes(b"concurrent")
            partial.write_bytes(b"new")
            with mock.patch.object(fetcher.os, "name", "posix"), mock.patch.object(fetcher.os, "rename") as rename:
                with self.assertRaises(FileExistsError):
                    fetcher._publish(partial, path)
                rename.assert_not_called()
            self.assertEqual(path.read_bytes(), b"concurrent")
            self.assertEqual(partial.read_bytes(), b"new")

    def test_lfs_identity_and_explicit_twenty_unique_ids(self):
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 123\n"
        self.assertEqual(fetcher.lfs_identity(pointer), ("a" * 64, 123))
        self.assertEqual(len(fetcher.IDS), 20)
        self.assertEqual(len(set(fetcher.IDS)), 20)
        with self.assertRaises(ValueError):
            fetcher.lfs_identity(b"not a pointer")


if __name__ == "__main__":
    unittest.main()
