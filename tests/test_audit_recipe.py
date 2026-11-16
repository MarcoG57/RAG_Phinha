"""gru11b (8 Nov audit fixes): ball speed over 5 frames, triangular overlapping windows and an
unweighted dead head; every default keeps gru11 byte for byte. Synthetic data only."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from possession.learning import features as F  # noqa: E402
from possession.learning import workflow  # noqa: E402
from test_tta_weights_variants import TINY_GRU, varied_match, varied_matches  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


def old_ball_speed(data):
    """The one-frame ball speed exactly as control_features computed it before ball_speed_frames."""
    ball = np.where(data.ball_observed[:, None], data.ball_xy, np.nan).astype(np.float64)
    speed = np.full(data.n_frames, np.nan)
    valid = (data.ball_observed[1:] & data.ball_observed[:-1]
             & (data.period_id[1:] == data.period_id[:-1]) & (data.period_id[1:] > 0))
    delta = np.linalg.norm(np.diff(ball * np.array(data.pitch), axis=0), axis=1) * data.fps
    speed[1:] = np.where(valid, delta, np.nan)
    return speed


class BallSpeedTests(unittest.TestCase):
    def test_default_ball_speed_is_unchanged(self):
        data = varied_match("one", 0).observations
        values, names = F.control_features(data)
        column = values[:, names.index("ball_speed_mps")]
        np.testing.assert_array_equal(np.isnan(column), np.isnan(old_ball_speed(data)))
        np.testing.assert_array_equal(column[~np.isnan(column)], old_ball_speed(data)[~np.isnan(column)])
        a, _ = F.onpitch_control_features(data)
        b, _ = F.onpitch_control_features(data, 1)
        np.testing.assert_array_equal(a, b)

    def test_five_frame_speed_and_only_that_column_change(self):
        data = varied_match("two", 1).observations
        base, names = F.onpitch_control_features(data)
        fast, names5 = F.onpitch_control_features_speed5(data)
        self.assertEqual(names, names5)
        changed = [names[j] for j in range(len(names)) if not np.array_equal(base[:, j], fast[:, j], equal_nan=True)]
        self.assertEqual(changed, ["ball_speed_mps"])
        k = 5
        ball = np.where(data.ball_observed[:, None], data.ball_xy, np.nan).astype(np.float64) * np.array(data.pitch)
        expected = np.full(data.n_frames, np.nan)
        valid = (data.ball_observed[k:] & data.ball_observed[:-k]
                 & (data.period_id[k:] == data.period_id[:-k]) & (data.period_id[k:] > 0))
        expected[k:] = np.where(valid, np.linalg.norm(ball[k:] - ball[:-k], axis=1) * data.fps / k, np.nan)
        column = fast[:, names.index("ball_speed_mps")]
        np.testing.assert_allclose(column, expected.astype(column.dtype), equal_nan=True)

class AuditSequenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        from possession.learning.sequence import SequenceHead
        rng = np.random.default_rng(0)
        n = 900
        self.x = rng.normal(size=(n, 3))
        self.period = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)].astype(np.int8)
        params = {"steps": 20, "hidden": 4, "layers": 1, "window_s": 4.0, "batch": 4}
        self.head = SequenceHead(params, device="cpu").fit([self.x], [self.x[:, 0] > 0], [np.ones(n, bool)],
                                                          [self.period], fps=25)

    def reference_triangular(self, head, overlap):
        """Independent re-statement of the Alt. 4 predictor (scripts/diagnosis/alt4_patches.py)."""
        import torch
        from possession.learning import sequence as SQ
        out = np.full(self.x.shape[0], 0.5)
        hop = max(1, head.window // overlap)
        with torch.no_grad():
            for idx in SQ._periods(self.period):
                x = SQ._encode(self.x[idx], head.centre, head.scale)
                n = len(x)
                starts = list(range(0, n - head.window + 1, hop)) if n > head.window else [0]
                if n > head.window and starts[-1] != n - head.window:
                    starts.append(n - head.window)
                total, weight = np.zeros(n), np.zeros(n)
                for start in starts:
                    piece = x[start:start + head.window]
                    w = np.minimum(np.arange(1, len(piece) + 1), np.arange(len(piece), 0, -1)).astype(float)
                    total[start:start + len(piece)] += w * head.model(torch.from_numpy(piece[None]))[0].numpy()
                    weight[start:start + len(piece)] += w
                frames = np.flatnonzero(self.period == self.period[idx[0]])
                out[frames] = np.interp(frames, idx, 1 / (1 + np.exp(-total / weight)))
        return out

    def test_default_and_explicit_half_overlap_predict_identically(self):
        before = self.head.predict(self.x, self.period)
        self.head.parameters = {**self.head.parameters, "window_overlap": 2, "triangular_average": False}
        np.testing.assert_array_equal(self.head.predict(self.x, self.period), before)

    def test_triangular_overlap_matches_the_evaluated_predictor(self):
        self.head.parameters = {**self.head.parameters, "window_overlap": 4, "triangular_average": True}
        actual = self.head.predict(self.x, self.period)
        np.testing.assert_allclose(actual, self.reference_triangular(self.head, 4), rtol=1e-12, atol=1e-12)
        self.assertTrue(np.isfinite(actual).all())

class DeadWeightsTests(unittest.TestCase):
    def record_fits(self, run):
        from possession.learning.sequence import SequenceHead
        calls, original = [], SequenceHead.fit

        def spy(head, features, targets, masks, periods, fps, **kwargs):
            calls.append("weights" in kwargs)
            return original(head, features, targets, masks, periods, fps, **kwargs)

        with mock.patch.object(SequenceHead, "fit", spy), \
                mock.patch("possession.learning.sequence.SequenceHead.__init__.__defaults__", (None, "cpu")):
            run()
        return calls  # per member: dead fit, then control fit

    def test_lomo_dead_head_drops_weights_only_when_configured(self):
        samples = varied_matches()
        inputs = workflow.head_inputs(samples, "gru11b", True)
        options = {"recipe": "gru11b", "periods": [m.observations.period_id for m in samples], "fps": 25}
        for flag, expected in ((True, [True, True]), (False, [False, True])):
            parameters = {**workflow.recipe_parameters("gru11b", TINY_GRU), "dead_loss_weights": flag}
            calls = self.record_fits(lambda: workflow.fit_predict_heads(inputs, [0, 2, 3, 4], [1], parameters, **options))
            self.assertEqual(calls, expected)

    def test_final_fit_follows_the_recipe(self):
        samples = varied_matches()
        calls = self.record_fits(lambda: workflow.fit_final(samples, "gru11b", parameter_overrides=TINY_GRU,
                                                            possession_weights=True))
        self.assertEqual(calls, [False, True])
        calls = self.record_fits(lambda: workflow.fit_final(samples, "gru11", parameter_overrides=TINY_GRU,
                                                            possession_weights=True))
        self.assertEqual(calls, [True, True])
