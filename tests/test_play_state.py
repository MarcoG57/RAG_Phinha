"""E2: evidencia de balón muerto, Viterbi vivo/muerto y combinación con E1; sintéticos."""
import importlib.util
import sys
import unittest
from pathlib import Path

if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("pandas") is None:
    raise unittest.SkipTest("numpy/pandas ausentes; instalar requirements.lock.txt (README.md)")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(path))
from possession.features.play_state import (  # noqa: E402
    PlayStateConfig, ball_speed_mps, dead_evidence, rolling_nanmean, team_speed_mps)
from synthetic_inputs import make_data  # noqa: E402

COST = 25.0
NO_SMOOTH = PlayStateConfig(smooth_s=0.04)  # ventana de 1 frame a 25 fps


def still_data(n, speed, ball=(0.5, 0.5)):
    """n frames con 3 jugadores a velocidad ``speed`` y el balón en ``ball``."""
    data = make_data([list(ball)] * n, [[[0.4, 0.5], [0.6, 0.5], [0.5, 0.6]]] * n, [[True] * 3] * n)
    data.speed[:] = speed
    return data


class FeatureTests(unittest.TestCase):
    def test_numeric_config_rejects_booleans(self):
        for name in PlayStateConfig.__dataclass_fields__:
            for value in (True, False, np.bool_(True)):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    PlayStateConfig(**{name: value})

    def test_rolling_nanmean_ignores_nan_and_keeps_empty_windows_nan(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan])
        out = rolling_nanmean(x, 3)
        self.assertAlmostEqual(out[1], 2.0)
        self.assertTrue(np.isnan(out[4]))
        np.testing.assert_array_equal(rolling_nanmean(x, 1), x)

    def test_rolling_nanmean_window_longer_than_sequence_preserves_shape(self):
        out = rolling_nanmean(np.arange(5, dtype=np.float64), 25)
        np.testing.assert_allclose(out, np.full(5, 2.0))

    def test_rolling_nanmean_empty_input_returns_empty(self):
        self.assertEqual(rolling_nanmean(np.array([], dtype=np.float64), 25).shape, (0,))

    def test_team_and_ball_speed(self):
        data = still_data(3, 2.0)
        data.observed[1] = False
        speed = team_speed_mps(data)
        self.assertEqual(speed[0], 2.0)
        self.assertTrue(np.isnan(speed[1]))
        data.ball_xy[1] = [0.51, 0.5]  # 1 m en x con pitch 100 m -> 25 m/s a 25 fps
        self.assertAlmostEqual(float(ball_speed_mps(data)[1]), 25.0, places=3)
        self.assertTrue(np.isnan(ball_speed_mps(data)[0]))

    def test_still_players_favour_dead_and_running_players_favour_play(self):
        self.assertGreater(dead_evidence(still_data(5, 0.5), NO_SMOOTH)[2], 0.3)
        self.assertLess(dead_evidence(still_data(5, 3.5), NO_SMOOTH)[2], -0.3)

    def test_ball_out_of_pitch_is_strong_dead_evidence(self):
        inside = dead_evidence(still_data(3, 2.5), NO_SMOOTH)[1]
        outside = dead_evidence(still_data(3, 2.5, ball=(1.05, 0.5)), NO_SMOOTH)[1]
        self.assertAlmostEqual(outside - inside, 1.0, places=6)
        margin = dead_evidence(still_data(3, 2.5, ball=(1.005, 0.5)), NO_SMOOTH)[1]
        self.assertAlmostEqual(margin, inside, places=6)

    def test_missing_ball_is_weak_and_fast_ball_counts_against_dead(self):
        base = dead_evidence(still_data(3, 0.5), NO_SMOOTH)
        data = still_data(3, 0.5)
        data.ball_observed[1] = False
        data.ball_xy[1] = np.nan
        self.assertAlmostEqual(dead_evidence(data, NO_SMOOTH)[1] - base[1], 0.15, places=6)
        data = still_data(3, 1.8)
        data.ball_xy[2] = [0.52, 0.5]  # 50 m/s
        neutral_base = dead_evidence(still_data(3, 1.8), NO_SMOOTH)
        self.assertAlmostEqual(neutral_base[2] - dead_evidence(data, NO_SMOOTH)[2], 0.5, places=6)

    def test_absence_without_additional_stop_evidence_does_not_make_dead(self):
        no_players = still_data(200, 1.8)
        no_players.observed[:] = False
        no_players.speed[:] = np.nan
        no_players.ball_observed[:] = False
        no_players.ball_xy[:] = np.nan
        absent = dead_evidence(no_players)
        np.testing.assert_allclose(absent, np.full(200, -PlayStateConfig().alive_bias))

        neutral_players = still_data(200, 1.8)
        neutral_players.ball_observed[:] = False
        neutral_players.ball_xy[:] = np.nan
        neutral = dead_evidence(neutral_players)
        np.testing.assert_allclose(neutral, np.full(200, -PlayStateConfig().alive_bias))

    def test_rest_speed_changes_do_not_affect_scored_periods(self):
        period = np.array([1] * 100 + [0] * 25 + [2] * 100, dtype=np.int8)
        baseline = still_data(225, 1.8)
        altered_rest = still_data(225, 1.8)
        baseline.period_id = period.copy()
        altered_rest.period_id = period.copy()
        altered_rest.speed[100:125] = 3.5

        scored = period > 0
        np.testing.assert_allclose(
            dead_evidence(baseline)[scored], dead_evidence(altered_rest)[scored]
        )

    def test_period_gap_and_contiguous_period_boundary_reset_features(self):
        data = still_data(5, 0.5)
        data.period_id = np.array([1, 1, 0, 2, 2], dtype=np.int8)
        data.speed[2] = 3.5
        data.speed[3:] = 3.5
        data.ball_xy[3] = [0.9, 0.5]  # discontinuity across the period-0 gap
        evidence = dead_evidence(data)
        self.assertGreater(evidence[1], 0.3)
        self.assertLess(evidence[3], -0.5)
        speed = ball_speed_mps(data)
        self.assertTrue(np.isnan(speed[2]))
        self.assertTrue(np.isnan(speed[3]))

        contiguous = still_data(4, 0.5)
        contiguous.period_id = np.array([1, 1, 2, 2], dtype=np.int8)
        contiguous.speed[2:] = 3.5
        contiguous.ball_xy[2] = [0.9, 0.5]
        boundary_speed = ball_speed_mps(contiguous)
        self.assertTrue(np.isnan(boundary_speed[2]))
        boundary_evidence = dead_evidence(contiguous)
        self.assertGreater(boundary_evidence[1], 0.3)
        self.assertLess(boundary_evidence[2], -0.5)

    def test_dead_evidence_shorter_than_smoothing_window_preserves_shape(self):
        evidence = dead_evidence(still_data(5, 0.5))
        self.assertEqual(evidence.shape, (5,))

    def test_config_is_validated(self):
        for bad in ({"out_weight": -1}, {"still_scale_mps": 0}, {"smooth_s": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PlayStateConfig(**bad)



if __name__ == "__main__":
    unittest.main()
