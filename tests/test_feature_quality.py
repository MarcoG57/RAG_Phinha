"""Feature quality regressions on synthetic observations only."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession.data.atd import AtdData, AtdMeta
from possession.learning.features import SHAPE_FEATURE_VERSION, team_shape_features


def empty_observations(match_id, source, fps, pitch, meta, n):
    slots = meta.n_slots
    return AtdData(match_id, source, fps, pitch, meta, np.arange(n, dtype=np.int32),
                   np.zeros(n, bool), np.zeros(n, np.int8), np.full((n, 2), np.nan, np.float32),
                   np.zeros(n, bool), np.full((n, slots, 2), np.nan, np.float32),
                   np.full((n, slots), np.nan, np.float32), np.zeros((n, slots), bool), periods_known=True)


class FeatureQualityTests(unittest.TestCase):
    def test_team_without_slots_keeps_unknown_shape_and_pressure(self):
        data = empty_observations("synthetic", "synthetic", 25, (100, 50),
                                  AtdMeta("H", "A", np.array(["H"])), 3)
        data.period_id[:] = 1
        data.xy[:] = [.1, .2]
        data.observed[:] = True
        values, names = team_shape_features(data)
        for name in ("H_nearest_opponent_m", "A_nearest_opponent_m", "A_front_m", "A_back_m"):
            self.assertTrue(np.isnan(values[:, names.index(name)]).all())

    def test_pressure_excludes_missing_players_and_keeps_unknown_opponents(self):
        data = empty_observations("synthetic", "synthetic", 25, (100, 50),
                                  AtdMeta("H", "A", np.array(["H", "H", "A"])), 3)
        data.period_id[:] = 1
        data.xy[0] = [[.1, .2], [np.nan, np.nan], [.4, .2]]
        data.xy[1] = [[.1, .2], [.2, .2], [np.nan, np.nan]]
        data.xy[2] = [[np.nan, np.nan], [np.nan, np.nan], [.4, .2]]
        data.observed[:] = np.isfinite(data.xy).all(axis=2)
        values, names = team_shape_features(data)
        for team in ("H", "A"):
            pressure = values[:, names.index(f"{team}_nearest_opponent_m")]
            self.assertAlmostEqual(float(pressure[0]), 30.0, places=4)
            self.assertTrue(np.isnan(pressure[1:]).all())
        self.assertEqual(SHAPE_FEATURE_VERSION, "team-shape-v2")


if __name__ == "__main__":
    unittest.main()
