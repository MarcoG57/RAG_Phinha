"""Submission export validation runs on tiny, synthetic prediction rows."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.evaluation.export import write_submission


class ExportTests(unittest.TestCase):
    @staticmethod
    def rows(labels=("H", "X")):
        return [{"frame": index, "label": label, "reason": "synthetic"}
                for index, label in enumerate(labels)]

    def test_writes_canonical_utf8_lf_and_returns_sha256(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "submission.csv"
            digest = write_submission(self.rows(), path, expected_frames=2)
            content = path.read_bytes()

        self.assertEqual(content, b"frame,label\n0,H\n1,X\n")
        self.assertNotIn(b"\r\n", content)
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "submission.csv"
            path.write_bytes(b"keep me\n")
            with self.assertRaises(FileExistsError):
                write_submission(self.rows(), path, expected_frames=2)
            self.assertEqual(path.read_bytes(), b"keep me\n")

    def test_invalid_rows_fail_before_a_final_file_is_created(self):
        invalid = (
            (self.rows(("H",)), 2),
            ([{"frame": 1, "label": "H"}, {"frame": 0, "label": "A"}], 2),
            ([{"frame": 0, "label": "H"}, {"frame": 0, "label": "A"}], 2),
            ([{"frame": 0, "label": "h"}, {"frame": 1, "label": "A"}], 2),
            ([{"frame": 0, "label": "H"}, {"frame": 1, "label": ""}], 2),
            (self.rows(), True),
        )
        for predictions, expected_frames in invalid:
            with self.subTest(predictions=predictions, expected_frames=expected_frames):
                with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                    path = Path(directory) / "submission.csv"
                    with self.assertRaises(ValueError):
                        write_submission(predictions, path, expected_frames)
                    self.assertFalse(path.exists())
                    self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
