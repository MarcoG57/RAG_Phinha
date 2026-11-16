"""Learning contracts tested exclusively on in-memory synthetic observations."""
import importlib.util
import sys
import unittest
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

if any(importlib.util.find_spec(name) is None for name in ("numpy", "pandas", "sklearn")):
    raise unittest.SkipTest("optional learning environment absent; requirements.lock.txt")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from possession.data.atd import AtdData, AtdMeta
from possession.learning.features import player_features
from possession.learning.features import control_features
from possession.learning.features import (nearest_event, swap_teams, team_shape_features,
                                          temporal_control_features, temporal_player_features, window_mean)
from possession.learning import workflow as workflow_module
from possession.learning.workflow import run_lomo
import tomllib

CONFIG = tomllib.loads((ROOT / "configs/model.toml").read_text(encoding="utf-8"))


@dataclass
class Match:
    """Consumer contract fixture; the production loader remains the sole producer."""
    match_id: str
    observations: AtdData
    reference: np.ndarray
    provenance: dict


def match(match_id="one", *, n=60, labels=None):
    period = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)].astype(np.int8)
    xy = np.tile([[[0.25, 0.5], [0.75, 0.5]]], (n, 1, 1)).astype(np.float32)
    data = AtdData(
        dataset_id="SYNTHETIC", source="synthetic", fps=25, pitch=(100.0, 50.0),
        meta=AtdMeta("Home", "Away", np.array(["H", "A"])),
        frame=np.arange(n, dtype=np.int32), record_present=np.ones(n, dtype=bool),
        period_id=period, ball_xy=xy[:, 0].copy(), ball_observed=np.ones(n, dtype=bool),
        xy=xy, speed=np.tile([[1.0, 3.0]], (n, 1)).astype(np.float32),
        observed=np.ones((n, 2), dtype=bool), periods_known=True,
    )
    reference = np.array(list("HAD") * (n // 3) + ["X"] * (n % 3)) if labels is None else np.array(labels)
    return Match(match_id, data, reference, {"evidence": "synthetic", "kind": "synthetic"})


class FeatureTests(unittest.TestCase):
    def test_features_preserve_frames_mask_absence_and_use_normalized_xy(self):
            data = match(n=6).observations
            data.observed[1, 0] = False
            data.xy[1, 0] = [999, 999]  # masked values cannot become observations
            data.speed[1, 0] = 999
            values, names = player_features(data)
            self.assertEqual(values.shape, (6, len(names)))
            self.assertEqual(values[0, names.index("H_x_mean_norm")], 0.25)
            self.assertEqual(values[0, names.index("A_speed_mean_mps")], 3.0)
            self.assertEqual(values[1, names.index("H_observed_fraction")], 0.0)
            self.assertTrue(np.isnan(values[1, names.index("H_x_mean_norm")]))
            self.assertTrue(np.isnan(values[1, names.index("H_speed_mean_mps")]))

    def test_ball_reference_and_metadata_are_not_features(self):
            original = match()
            changed = deepcopy(original)
            changed.reference[:] = "D"
            changed.observations.ball_xy[:] = 99
            changed.observations.ball_observed[:] = False
            changed.observations.dataset_id = "not-a-feature"
            changed.observations.provenance = {"gt": ["D"] * 60}
            a, names = player_features(original.observations)
            b, other_names = player_features(changed.observations)
            self.assertEqual(names, other_names)
            np.testing.assert_array_equal(a, b)

    def test_changing_other_period_cannot_change_features(self):
            data = match().observations
            a, _ = player_features(data)
            data.xy[data.period_id == 2] *= 2
            data.speed[data.period_id == 2] = 99
            b, _ = player_features(data)
            np.testing.assert_array_equal(a[:30], b[:30])

    def test_ball_speed_resets_at_period_and_masks_missing_ball(self):
            data = match(n=6).observations
            data.ball_xy[:, 0] = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
            data.ball_observed[4] = False
            values, names = control_features(data)
            speed = values[:, names.index("ball_speed_mps")]
            self.assertAlmostEqual(speed[1], 250, places=3)
            self.assertTrue(np.isnan(speed[[0, 3, 4, 5]]).all())

class TemporalFeatureTests(unittest.TestCase):
    def test_window_mean_ignores_nan_and_stops_at_the_period(self):
            values = np.array([1.0, np.nan, 3.0, 10.0, 20.0])
            period = np.array([1, 1, 1, 2, 2])
            out = window_mean(values, period, 3)
            np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 15.0, 15.0])

    def test_nearest_event_counts_seconds_within_each_period(self):
            mask = np.array([False, True, False, False, True, False])
            period = np.array([1, 1, 1, 1, 2, 2])
            last, nxt, since, until = nearest_event(mask, period, 2.0)
            np.testing.assert_array_equal(last, [-1, 1, 1, 1, 4, 4])
            np.testing.assert_array_equal(nxt, [1, 1, -1, -1, 4, -1])
            np.testing.assert_allclose(since, [np.nan, 0, .5, 1, 0, .5])
            np.testing.assert_allclose(until, [.5, 0, np.nan, np.nan, 0, np.nan])

    def test_dead_features_still_ignore_the_ball_and_other_periods(self):
            original = match().observations
            changed = deepcopy(original)
            changed.ball_xy[:] = 99
            changed.ball_observed[:] = False
            changed.xy[changed.period_id == 2] *= 2
            changed.speed[changed.period_id == 2] = 99
            a, names = temporal_player_features(original)
            b, _ = temporal_player_features(changed)
            self.assertEqual(a.shape, (60, len(names)))
            np.testing.assert_array_equal(a[:30], b[:30])

    def test_ball_context_carries_the_last_and_next_sighting(self):
            data = match(n=10).observations
            data.ball_observed[3:7] = False
            values, names = temporal_control_features(data)
            self.assertEqual(values.dtype, np.float32)
            col = {name: values[:, names.index(name)] for name in names}
            np.testing.assert_allclose(col["since_ball_s"][[2, 3, 4]], [0, .04, .08], atol=1e-6)
            np.testing.assert_allclose(col["until_ball_s"][[1, 2]], [0, 0])
            self.assertTrue(np.isnan(col["since_ball_s"][5]))  # period 2 starts at frame 5, ball unseen until 7
            self.assertEqual(col["margin_at_last_ball_m"][4], 50.0)  # H player on the ball, A 50 m away
            self.assertEqual(col["last_close_team"][4], 1.0)

    def test_swap_teams_mirrors_the_control_context(self):
            data = match(n=10).observations
            a, names = temporal_control_features(data)
            b, _ = temporal_control_features(swap_teams(data))
            i = names.index("margin_m_w1s")
            np.testing.assert_allclose(b[:, i], -a[:, i])
            self.assertEqual(b[0, names.index("last_close_team")], -1.0)
            twice = swap_teams(swap_teams(data))
            np.testing.assert_array_equal(twice.meta.slot_team, data.meta.slot_team)
            np.testing.assert_allclose(twice.xy, data.xy, atol=1e-6)

class TeamShapeTests(unittest.TestCase):
    def test_attack_direction_advance_pressure_and_forward_speed(self):
            data = match(n=20).observations
            data.xy[:10, 0, 0] = .25 + .01 * np.arange(10)  # home moves 1 m per frame towards x = 1
            values, names = team_shape_features(data)
            col = {name: values[:, names.index(name)] for name in names}
            self.assertAlmostEqual(float(col["A_advance_m"][15]), -25.0, places=4)  # away stands at x = .75, attacks to 0
            self.assertAlmostEqual(float(col["H_nearest_opponent_m"][15]), 50.0, places=4)
            np.testing.assert_allclose(col["H_forward_speed_w1s"][2:8], 25.0, rtol=1e-5)

    def test_swapping_teams_exchanges_the_shape_and_ignores_the_ball(self):
            data = match(n=20).observations
            a, names = team_shape_features(data)
            b, _ = team_shape_features(swap_teams(data))
            for name in names:
                other = name.replace("H_", "@").replace("A_", "H_").replace("@", "A_")
                sign = -1 if name.endswith(("diff_m", "diff", "diff_w3s")) else 1
                np.testing.assert_allclose(b[:, names.index(other)], sign * a[:, names.index(name)], atol=1e-4, err_msg=name)
            changed = deepcopy(data)
            changed.ball_xy[:] = 99
            changed.ball_observed[:] = False
            np.testing.assert_array_equal(team_shape_features(changed)[0], a)

class OnPitchTests(unittest.TestCase):
    def test_availability_counts_against_eleven_players_not_roster_slots(self):
            from possession.learning.features import onpitch_player_features, shape_player_features
            data = match(n=20).observations  # one slot per team, always observed
            base, names = shape_player_features(data)
            values, same = onpitch_player_features(data)
            self.assertEqual(names, same)
            i = names.index("H_observed_fraction")
            self.assertEqual(base[0, i], 1.0)
            self.assertAlmostEqual(float(values[0, i]), 1 / 11, places=6)
            j = names.index("H_speed_mean_mps")
            np.testing.assert_array_equal(values[:, j], base[:, j])

class WorkflowTests(unittest.TestCase):
    def test_average_logits_is_the_sigmoid_of_the_mean_log_odds(self):
            a, b = np.array([.9, .2, 1.0]), np.array([.6, .5, .0])
            logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
            expected = 1 / (1 + np.exp(-(logit(a) + logit(b)) / 2))
            np.testing.assert_allclose(workflow_module.average_logits([a, b]), expected, rtol=1e-12)
            np.testing.assert_array_equal(workflow_module.average_logits([a]), a)  # one member: unchanged
            with self.assertRaisesRegex(ValueError, "aligned"):
                workflow_module.average_logits([a, b[:2]])

    def test_held_out_reference_never_affects_gru_prediction(self):
        matches = [match(name) for name in ("one", "two", "three")]
        tiny = {"steps": 4, "hidden": 4, "layers": 1, "window_s": 2.0, "batch": 4}
        with mock.patch("possession.learning.sequence.SequenceHead.__init__.__defaults__", (None, "cpu")):
            first = run_lomo(matches, CONFIG, recipe="gru11", parameter_overrides=tiny)
            matches[0].reference[:] = "X"
            second = run_lomo(matches, CONFIG, recipe="gru11", parameter_overrides=tiny)
        a = next(f for f in first if f["held_out_match_id"] == "one")
        b = next(f for f in second if f["held_out_match_id"] == "one")
        self.assertNotIn("one", a["train_match_ids"])
        for key in ("probability_H", "probability_D"):
            np.testing.assert_array_equal(a["predictions"][0][key], b["predictions"][0][key])
        for name, labels in a["predictions"][0]["labels"].items():
            np.testing.assert_array_equal(labels, b["predictions"][0]["labels"][name])

    def test_gru_seed_ensemble_averages_member_logits_and_takes_overrides(self):
            samples = []
            for name in ("one", "two", "three"):
                sample = match(name, n=300, labels=["H"] * 100 + ["D"] * 100 + ["A"] * 100)
                sample.observations.speed[100:200] = 0
                samples.append(sample)
            tiny = {"steps": 8, "hidden": 4, "layers": 1, "window_s": 2.0, "batch": 4}
            seen = []
            from possession.learning.sequence import SequenceHead
            real_init = SequenceHead.__init__
            def cpu_init(self, parameters=None, device=None):
                seen.append(dict(parameters))
                real_init(self, parameters, device="cpu")
            with mock.patch.object(SequenceHead, "__init__", cpu_init):
                single = [run_lomo(samples, CONFIG, recipe="gru11", parameter_overrides=tiny, model_seed=s) for s in (1, 2)]
                one = run_lomo(samples, CONFIG, recipe="gru11", parameter_overrides=tiny, ensemble_seeds=[1])
                both = run_lomo(samples, CONFIG, recipe="gru11", parameter_overrides=tiny, ensemble_seeds=[1, 2])
            self.assertTrue(all(p["hidden"] == 4 and p["steps"] == 8 for p in seen))
            self.assertEqual([p["seed"] for p in seen[-12:]], [1, 1, 2, 2] * 3)
            logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
            for k, fold in enumerate(both):
                for key in ("probability_H", "probability_D"):
                    a, b = (run[k]["predictions"][0][key] for run in single)
                    self.assertFalse(np.array_equal(a, b))  # the seeds give different members
                    np.testing.assert_array_equal(one[k]["predictions"][0][key], a)
                    np.testing.assert_allclose(fold["predictions"][0][key],
                                               1 / (1 + np.exp(-(logit(a) + logit(b)) / 2)), rtol=1e-12)
