"""BiGRU heads: CPU inference after save/load is exact and repeatable; synthetic data only."""
import importlib.util
import sys
import tempfile
import hashlib
import unittest
from unittest import mock
from pathlib import Path

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("PyTorch ausente; instalar requirements.lock.txt")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.learning.sequence import SequenceHead  # noqa: E402

SMALL = {"steps": 40, "hidden": 8, "layers": 1, "window_s": 4.0, "batch": 4}


def data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x[::7, 1] = np.nan
    y = x[:, 0] > 0
    period = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)].astype(np.int8)
    return x, y, np.ones(n, bool), period


class SequenceHeadTests(unittest.TestCase):
    def test_untrusted_checkpoint_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrusted.pt"
            path.write_bytes(b"untrusted checkpoint")
            with mock.patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    SequenceHead.load(path, trusted_sha256="0" * 64)
                load.assert_not_called()


    def test_cpu_inference_after_save_and_load_is_exact(self):
        torch.set_num_threads(1)
        x, y, mask, period = data()
        head = SequenceHead(SMALL, device="cpu").fit([x], [y], [mask], [period], fps=25)
        first = head.predict(x, period)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            head.save(path)
            loaded = SequenceHead.load(path, trusted_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_array_equal(loaded.predict(x, period), first)
            np.testing.assert_array_equal(SequenceHead.load(path, trusted_sha256=hashlib.sha256(path.read_bytes()).hexdigest()).predict(x, period), first)
        self.assertTrue(((first >= 0) & (first <= 1)).all())

    def test_training_on_cpu_repeats_exactly_and_frames_outside_periods_are_neutral(self):
        torch.set_num_threads(1)
        x, y, mask, period = data()
        period[:10] = 0
        a = SequenceHead(SMALL, device="cpu").fit([x], [y], [mask], [period], fps=25).predict(x, period)
        b = SequenceHead(SMALL, device="cpu").fit([x], [y], [mask], [period], fps=25).predict(x, period)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:10], 0.5)

    def test_cpu_fit_records_environment_and_restores_caller_settings(self):
        x, y, mask, period = data(120)
        before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
        head = SequenceHead({**SMALL, "steps": 4}, device="cpu").fit([x], [y], [mask], [period], fps=25)
        self.assertEqual((torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()), before)
        self.assertEqual(head.environment["device"], "cpu")
        self.assertEqual(head.environment["threads"], 1)
        self.assertTrue(head.environment["deterministic_algorithms"])
        self.assertEqual(head.environment["torch"], torch.__version__)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            head.save(path)
            self.assertEqual(SequenceHead.load(path, trusted_sha256=hashlib.sha256(path.read_bytes()).hexdigest()).environment, head.environment)

    def test_normalisation_uses_only_known_training_rows_and_prediction_resets_each_period(self):
        x, y, mask, period = data(120)
        mask[1::2] = False
        x[~mask] = 1e9
        head = SequenceHead({**SMALL, "steps": 4}, device="cpu").fit([x], [y], [mask], [period], fps=25)
        np.testing.assert_allclose(head.centre, np.nanmean(x[mask], axis=0))
        first = head.predict(x, period)
        changed = x.copy()
        changed[period == 1] = -1e8
        np.testing.assert_array_equal(first[period == 2], head.predict(changed, period)[period == 2])


if __name__ == "__main__":
    unittest.main()
