"""Acquisition duplicates must fail before training, even when aliases have distinct ids."""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_final


class TrainingAcquisitionTests(unittest.TestCase):
    def duplicate_manifest(self, folder):
        (folder / "same.json").write_bytes((ROOT / "tests/fixtures/external/one.json").read_bytes())
        path = folder / "aliases.toml"
        path.write_text("".join(f'[[matches]]\nid = "{mid}"\nformat = "synthetic"\npath = "same.json"\n'
                                for mid in ("a", "b", "c")), encoding="utf-8")
        return path

    def test_final_training_audits_before_fitting_and_keeps_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = self.duplicate_manifest(folder)
            with (mock.patch.object(train_final, "MODELS", folder / "models"),
                  mock.patch("possession.learning.workflow.fit_final") as fit,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                code = train_final.main(["--manifest", str(path), "--recipe", "gru11", "--control-cost", "12.5",
                                         "--dead-cost", "25", "--name", "duplicate"])
                self.assertEqual(code, 1)
                fit.assert_not_called()
                self.assertFalse((folder / "models").exists())


if __name__ == "__main__":
    unittest.main()
