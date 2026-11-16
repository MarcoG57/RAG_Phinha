"""Inference-time augmentation has the independently specified logit mean."""
import sys
from dataclasses import dataclass
from unittest import mock
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession.learning import workflow
from possession.data.atd import AtdData, AtdMeta
from test_learning import CONFIG

TINY_GRU = {"steps": 6, "hidden": 4, "layers": 1, "window_s": 2.0, "batch": 4}

def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)

class TestTimeSwapTests(unittest.TestCase):
    def test_swap_average_is_the_manual_logit_formula(self):
            rng = np.random.default_rng(3)
            ph, pd, sh, sd = rng.uniform(0, 1, size=(4, 50))
            ph[:2], sh[:2] = [0.0, 1.0], [1.0, 0.0]  # clipped like the decoder evidence
            h, d = workflow.swap_average(ph, pd, sh, sd)
            np.testing.assert_allclose(h, 1 / (1 + np.exp(-(logit(ph) + logit(1 - sh)) / 2)), rtol=1e-12)
            np.testing.assert_allclose(d, 1 / (1 + np.exp(-(logit(pd) + logit(sd)) / 2)), rtol=1e-12)
            # A perfectly mirrored model leaves the prediction unchanged (away from the clip).
            np.testing.assert_allclose(workflow.swap_average(ph, pd, 1 - ph, pd)[0][2:], ph[2:], rtol=1e-9)


@dataclass
class Match:
    match_id: str
    observations: AtdData
    reference: np.ndarray
    provenance: dict

def varied_match(match_id: str, seed: int, n: int = 600) -> Match:
    """Three players per team; the team in possession stands next to the ball."""
    rng = np.random.default_rng(seed)
    period = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)].astype(np.int8)
    labels, team = [], "H"
    while len(labels) < n:
        length = int(rng.integers(10, 80))
        kind = "D" if rng.random() < .2 else team
        labels += [kind] * length
        team = "A" if team == "H" else "H"
    reference = np.array(labels[:n])
    reference[rng.random(n) < .03] = "X"
    ball = np.clip(.5 + np.cumsum(rng.normal(0, .01, size=(n, 2)), axis=0), .05, .95)
    xy = rng.uniform(.1, .9, size=(n, 6, 2))
    owner = np.where(reference == "H", 0, 3)
    held = np.isin(reference, ["H", "A"])
    xy[held, owner[held]] = ball[held] + rng.normal(0, .005, size=(held.sum(), 2))
    speed = np.where((reference == "D")[:, None], rng.uniform(0, .5, (n, 6)), rng.uniform(1, 5, (n, 6)))
    observed = rng.random((n, 6)) > .05
    data = AtdData(
        dataset_id="SYNTHETIC", source="synthetic", fps=25, pitch=(105.0, 68.0),
        meta=AtdMeta("Home", "Away", np.array(["H", "H", "H", "A", "A", "A"])),
        frame=np.arange(n, dtype=np.int32), record_present=np.ones(n, dtype=bool),
        period_id=period, ball_xy=ball.astype(np.float32), ball_observed=rng.random(n) > .2,
        xy=xy.astype(np.float32), speed=speed.astype(np.float32), observed=observed, periods_known=True)
    return Match(match_id, data, reference, {"evidence": "synthetic", "kind": "synthetic"})

def varied_matches() -> list[Match]:
    return [varied_match(name, seed) for seed, name in enumerate(("one", "two", "three", "four", "extra"))]

class PossessionWeightTests(unittest.TestCase):
    def test_spell_weights_on_a_toy_reference(self):
            # fps 4: a spell of 32 frames (8 s) weighs 1.
            parts = [("H" * 4, 0), ("H" * 4 + "X" * 2 + "H" * 4 + "A" * 64 + "D" * 3 + "A" * 200 + "H" * 2, 1),
                     ("H" * 16 + "X", 2)]
            reference = np.array(list("".join(text for text, _ in parts)))
            period = np.concatenate([np.full(len(text), p, dtype=np.int8) for text, p in parts])
            weights = workflow.spell_weights(reference, period, 4)
            self.assertEqual(weights.dtype, np.float32)
            expected = np.r_[np.ones(4),          # period 0: never trained
                             np.full(4, 4.0),     # 8 H frames, X skipped: 32 / 8 = 4
                             np.ones(2),          # X
                             np.full(4, 4.0),
                             np.full(64, .5),     # 16 s
                             np.ones(3),          # D ends the spell
                             np.full(200, .25),   # 50 s, clipped
                             np.full(2, 4.0),     # 2 frames, clipped; the period ends the spell
                             np.full(16, 2.0),    # period 2: a new spell of 4 s
                             np.ones(1)]
            np.testing.assert_array_equal(weights, expected.astype(np.float32))
            self.assertEqual(workflow.spell_weights(np.array(list("HHXHH")), np.ones(5, np.int8), 1).tolist(),
                             [2.0, 2.0, 1.0, 2.0, 2.0])

    def test_gru_loss_mask_is_multiplied_by_the_weights(self):
            from possession.learning.sequence import SequenceHead
            rng = np.random.default_rng(0)
            x = rng.normal(size=(400, 3))
            y, mask = x[:, 0] > 0, np.ones(400, bool)
            period = np.r_[np.ones(200), np.full(200, 2)].astype(np.int8)
            fit = lambda **kw: SequenceHead(TINY_GRU, device="cpu").fit([x], [y], [mask], [period], 25, **kw)  # noqa: E731
            base = fit().predict(x, period)
            np.testing.assert_array_equal(fit(weights=[np.ones(400, np.float32)]).predict(x, period), base)
            weights = np.where(y, 4.0, .25).astype(np.float32)
            self.assertFalse(np.array_equal(fit(weights=[weights]).predict(x, period), base))
            with self.assertRaisesRegex(ValueError, "align"):
                fit(weights=[weights, weights])

            seen = []
            real = SequenceHead.fit

            def spy(head, features, targets, masks, periods, fps, weights=None):
                seen.append(None if weights is None else len(weights))
                return real(head, features, targets, masks, periods, fps, weights=weights)

            samples = varied_matches()
            inputs = workflow.head_inputs(samples, "gru11", possession_weights=True)
            with mock.patch.object(SequenceHead, "fit", autospec=True, side_effect=spy), mock.patch(
                    "possession.learning.sequence.SequenceHead.__init__.__defaults__", (None, "cpu")):
                workflow.fit_predict_heads(inputs, [0, 2, 3, 4], [1], workflow.recipe_parameters("gru11", TINY_GRU),
                                           recipe="gru11", periods=[m.observations.period_id for m in samples], fps=25)
            self.assertEqual(seen, [4, 8])

    def test_run_lomo_and_final_pass_weights_to_each_fit(self):
        from possession.learning.sequence import SequenceHead
        samples = varied_matches()
        real = SequenceHead.fit
        seen = []
        def spy(head, features, targets, masks, periods, fps, weights=None):
            seen.append((len(features), None if weights is None else [w.copy() for w in weights]))
            return real(head, features, targets, masks, periods, fps, weights=weights)
        with (mock.patch.object(SequenceHead, "fit", autospec=True, side_effect=spy),
              mock.patch("possession.learning.sequence.SequenceHead.__init__.__defaults__", (None, "cpu"))):
            folds = workflow.run_lomo(samples, CONFIG, recipe="gru11", train_only=frozenset({"extra"}),
                                      possession_weights=True, parameter_overrides=TINY_GRU)
            fitted = workflow.fit_final(samples, "gru11", possession_weights=True, parameter_overrides=TINY_GRU)
        self.assertEqual(len(folds), 4)
        self.assertEqual([n for n, _ in seen], [4, 8] * 4 + [5, 10])
        for _, weights in seen:
            self.assertIsNotNone(weights)
            self.assertTrue(any(np.any(w != 1) for w in weights))
        self.assertTrue(fitted["possession_weights"])
