"""Synthetic CSV tests; no match material is read."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_submission
from check_submission import strict_preflight


class SubmissionTests(unittest.TestCase):
    def check_text(self, text, expected=2):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text(text, encoding="utf-8", newline="")
            strict_preflight(path, expected)

    def test_extra_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "header"):
            self.check_text("frame,label,index\n0,H,0\n1,A,1\n")

    def test_canonical_csv_and_crlf_are_accepted(self):
        self.check_text("frame,label\r\n0,H\r\n1,X\r\n")

    def test_alignment_and_malformed_rows_are_rejected(self):
        for text in ("frame,label\n1,H\n2,A\n", "frame,label\n0,H\n0,A\n",
                     "frame,label\n0,H\n1.0,A\n", "frame,label\n0,H\n1,A,extra\n",
                     "frame,label\n0,H\n", "frame,label\n0,H\n1,A\n2,D\n",
                     "frame,label\n0,H\n\n1,A\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check_text(text)

    def test_noncanonical_and_empty_labels_are_rejected(self):
        for label in ("", "h", " H ", "N/A", "Z"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.check_text(f"frame,label\n0,H\n1,{label}\n")

    def test_invalid_expected_count_is_rejected(self):
        for expected in (0, None):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.check_text("frame,label\n", expected=expected)

    def test_reader_and_preflight_accept_the_same_c4_inputs(self):
        cases = (
            ("frame,label\r\n0,H\r\n1,X\r\n", 2, True),
            ("frame,label\n0,H\n", 2, False),
            ("frame,label\n0,H\n1,A\n2,D\n", 2, False),
            ("frame,label,index\n0,H,0\n1,A,1\n", 2, False),
            ("frame,label\n1,H\n2,A\n", 2, False),
            ("frame,label\n0,H\n0,A\n", 2, False),
            ("frame,label\n0,H\n1,Home\n", 2, False),
            ('frame,label\n0,"H\n1,A\n', 2, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            for text, expected, valid in cases:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8", newline="")
                    preflight_error = self._error_from(lambda: strict_preflight(path, expected))
                    read_error = self._error_from(
                        lambda: check_submission.read_submission_labels(path, expected)
                    )
                    self.assertEqual(preflight_error, read_error)
                    self.assertEqual(preflight_error is None, valid)

    def test_shared_c4_reader_opens_the_csv_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text("frame,label\n0,H\n1,X\n", encoding="utf-8", newline="")
            original_open = Path.open
            opened = []

            def counted_open(candidate, *args, **kwargs):
                if candidate == path:
                    opened.append(candidate)
                return original_open(candidate, *args, **kwargs)

            with patch.object(Path, "open", new=counted_open):
                self.assertEqual(check_submission.read_submission_labels(path, 2), ["H", "X"])
            self.assertEqual(opened, [path])

    @staticmethod
    def _error_from(operation):
        try:
            operation()
        except (ValueError, csv.Error) as error:
            return type(error), str(error)
        return None


if __name__ == "__main__":
    unittest.main()
