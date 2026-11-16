"""SkillCorner adapter on tiny synthetic files; no real match data."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("numpy ausente; instalar requirements.lock.txt (README.md)")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.data.external import iter_matches  # noqa: E402
from possession.data.skillcorner import load_skillcorner  # noqa: E402

PHASES_HEADER = "index,match_id,frame_start,frame_end,period,team_in_possession_id\n"


def write_match(folder: Path, *, ball_detected=True):
    """21 source frames at 10 fps: period 1 on frames 0..20, H phase 2..8, A phase 8..14, D after."""
    match = {"id": 7, "home_team": {"id": 1, "name": "Home"}, "away_team": {"id": 2, "name": "Away"},
             "pitch_length": 100, "pitch_width": 50,
             "match_periods": [{"period": 1, "start_frame": 0, "end_frame": 20}],
             "players": [{"id": 11, "team_id": 1}, {"id": 21, "team_id": 2}]}
    (folder / "m.json").write_text(json.dumps(match), encoding="utf-8")
    with (folder / "t.jsonl").open("w", encoding="utf-8") as handle:
        for frame in range(21):
            players = [{"player_id": 11, "x": float(frame), "y": 0.0, "is_detected": True},
                       {"player_id": 21, "x": -10.0, "y": 5.0, "is_detected": frame % 2 == 0}]
            ball = {"x": float(frame), "y": 0.0, "z": 0.0,
                    "is_detected": ball_detected and frame != 10}
            handle.write(json.dumps({"frame": frame, "period": 1, "ball_data": ball,
                                     "possession": {"player_id": None, "group": None},
                                     "player_data": players}) + "\n")
    (folder / "p.csv").write_text(PHASES_HEADER + "0,7,2,8,1,1\n1,7,8,14,1,2\n", encoding="utf-8")
    return folder / "m.json", folder / "t.jsonl", folder / "p.csv"


class SkillCornerTests(unittest.TestCase):
    def test_nonfinite_ball_coordinates_are_missing_even_when_flagged_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
            rows[5]["ball_data"]["x"] = float("nan")
            paths[1].write_text("\n".join(json.dumps(row) for row in rows))
            data = load_skillcorner(*paths, match_id="SC-7").observations
        self.assertFalse(data.ball_observed[12])  # nearest source frame 5
        self.assertTrue(np.isfinite(data.ball_xy[data.ball_observed]).all())

    def test_invalid_period_bounds_and_repeated_periods_are_rejected(self):
        invalid = [
            [{"period": 1, "start_frame": -1, "end_frame": 20}],
            [{"period": 1, "start_frame": 0, "end_frame": 22}],
            [{"period": 1, "start_frame": 10, "end_frame": 9}],
            [{"period": 5, "start_frame": 0, "end_frame": 20}],
            [{"period": 1, "start_frame": 0, "end_frame": 9},
             {"period": 1, "start_frame": 10, "end_frame": 20}],
            [{"period": 1, "start_frame": 0, "end_frame": 10},
             {"period": 2, "start_frame": 10, "end_frame": 20}],
        ]
        for periods in invalid:
            with self.subTest(periods=periods), tempfile.TemporaryDirectory() as tmp:
                paths = write_match(Path(tmp))
                raw = json.loads(paths[0].read_text())
                raw["match_periods"] = periods
                paths[0].write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, "period"):
                    load_skillcorner(*paths, match_id="SC-7")

    def test_missing_terminal_period_endpoint_is_audited_without_inventing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            raw = json.loads(paths[0].read_text())
            raw["match_periods"] = [{"period": 1, "start_frame": 0, "end_frame": 8},
                                    {"period": 2, "start_frame": 10, "end_frame": 21}]
            paths[0].write_text(json.dumps(raw))
            paths[2].write_text(PHASES_HEADER)
            result = load_skillcorner(*paths, match_id="SC-7")
        self.assertEqual(result.observations.n_frames, 51)  # still source frames 0..20
        self.assertEqual(result.observations.period_id[-1], 2)
        adjustments = result.provenance["period_boundary_adjustments"]
        self.assertEqual(adjustments["clipped_period_count"], 1)
        self.assertEqual(adjustments["missing_terminal_source_frames"], 1)
        self.assertEqual(result.observations.provenance["period_boundary_adjustments"], adjustments)

    def test_phase_intervals_are_half_open_bounded_and_conflict_checked(self):
        for phases in ("0,7,-2,8,1,1\n", "0,7,2,22,1,1\n", "0,7,8,2,1,1\n",
                       "0,7,2,8,2,1\n", "0,7,2,9,1,1\n1,7,8,14,1,2\n"):
            with self.subTest(phases=phases), tempfile.TemporaryDirectory() as tmp:
                paths = write_match(Path(tmp))
                paths[2].write_text(PHASES_HEADER + phases)
                with self.assertRaisesRegex(ValueError, "phase"):
                    load_skillcorner(*paths, match_id="SC-7")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            paths[2].write_text(PHASES_HEADER + "0,7,2,9,1,1\n1,7,8,14,1,1\n")
            result = load_skillcorner(*paths, match_id="SC-7")
            self.assertEqual(result.reference[20], "H")
            self.assertEqual(result.reference[35], "D")  # source frame 14, excluded stop

    def test_phase_tail_beyond_its_period_is_clipped_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            raw = json.loads(paths[0].read_text())
            raw["match_periods"] = [{"period": 1, "start_frame": 0, "end_frame": 8},
                                    {"period": 2, "start_frame": 10, "end_frame": 20}]
            paths[0].write_text(json.dumps(raw))
            # Phase declares period 1; its tail runs through the break into period 2.
            # The period metadata, independently of labels, defines the valid clock.
            paths[2].write_text(PHASES_HEADER + "0,7,7,13,1,1\n1,7,12,15,2,2\n")
            result = load_skillcorner(*paths, match_id="SC-7")
        nearest = np.rint(np.arange(51) * .4).astype(int)
        ref = result.reference
        self.assertTrue((ref[(nearest >= 7) & (nearest < 9)] == "H").all())
        self.assertTrue((ref[nearest == 9] == "X").all())
        self.assertTrue((ref[(nearest >= 10) & (nearest < 12)] == "D").all())
        self.assertTrue((ref[(nearest >= 12) & (nearest < 15)] == "A").all())
        adjustments = result.provenance["phase_boundary_adjustments"]
        self.assertEqual(adjustments["clipped_phase_count"], 1)
        self.assertEqual(adjustments["clipped_source_frames"], 4)

    def test_phase_start_outside_declared_period_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            raw = json.loads(paths[0].read_text())
            raw["match_periods"] = [{"period": 1, "start_frame": 0, "end_frame": 8},
                                    {"period": 2, "start_frame": 10, "end_frame": 20}]
            paths[0].write_text(json.dumps(raw))
            paths[2].write_text(PHASES_HEADER + "0,7,10,13,1,1\n")
            with self.assertRaisesRegex(ValueError, "phase"):
                load_skillcorner(*paths, match_id="SC-7")

    def test_interpolation_and_speeds_do_not_use_next_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            raw = json.loads(paths[0].read_text())
            raw["match_periods"] = [{"period": 1, "start_frame": 0, "end_frame": 9},
                                    {"period": 2, "start_frame": 10, "end_frame": 20}]
            paths[0].write_text(json.dumps(raw))
            paths[2].write_text(PHASES_HEADER)
            rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
            for row in rows[10:]:
                row["player_data"][0]["x"] += 40
                row["ball_data"]["x"] += 40
                row["ball_data"]["is_detected"] = True
            paths[1].write_text("\n".join(json.dumps(row) for row in rows))
            result = load_skillcorner(*paths, match_id="SC-7")
        data = result.observations
        self.assertEqual(data.period_id[23], 1)  # source 9.2: nearest 9, next belongs to 2
        self.assertAlmostEqual(float(data.xy[23, 0, 0]), .59, places=6)
        self.assertAlmostEqual(float(data.ball_xy[23, 0]), .59, places=6)
        self.assertTrue(np.isnan(data.speed[23:27, 0]).all())
        self.assertEqual(data.period_id[24], 2)  # source 9.6: nearest 10, previous belongs to 1
        self.assertAlmostEqual(float(data.xy[24, 0, 0]), 1.0, places=6)
        self.assertIn("coverage", result.provenance)
        self.assertTrue(result.provenance["includes_extrapolated_players"])

    def test_gap_between_periods_keeps_observations_and_reference_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            raw = json.loads(paths[0].read_text())
            raw["match_periods"] = [{"period": 1, "start_frame": 0, "end_frame": 8},
                                    {"period": 2, "start_frame": 11, "end_frame": 20}]
            paths[0].write_text(json.dumps(raw))
            paths[2].write_text(PHASES_HEADER)
            result = load_skillcorner(*paths, match_id="SC-7", players="detected")
        outside = result.observations.period_id == 0
        self.assertTrue(outside.any())
        self.assertTrue((result.reference[outside] == "X").all())
        self.assertFalse(result.observations.observed[outside].any())
        self.assertFalse(result.observations.ball_observed[outside].any())
        self.assertTrue(np.isnan(result.observations.speed[outside]).all())
        self.assertFalse(result.provenance["includes_extrapolated_players"])
        self.assertLess(result.provenance["coverage"]["target_observed_player_samples"],
                        result.provenance["coverage"]["target_player_samples"])

    def test_resamples_to_25_fps_with_phase_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = load_skillcorner(*write_match(Path(tmp)), match_id="SC-7")
        data, ref = match.observations, match.reference
        self.assertEqual((data.fps, data.n_frames), (25, 51))  # (21 - 1) * 2.5 + 1
        nearest = np.rint(np.arange(51) * 0.4)  # nearest source frame of each target frame
        expected = np.where((nearest >= 2) & (nearest < 8), "H", np.where((nearest >= 8) & (nearest < 14), "A", "D"))
        np.testing.assert_array_equal(ref, expected)
        self.assertTrue((data.period_id == 1).all())
        # Linear interpolation of the home player at x = source frame (metres) from the centre.
        np.testing.assert_allclose(data.xy[3, 0, 0], 1.2 / 100 + .5, atol=1e-6)
        self.assertAlmostEqual(float(np.nanmedian(data.speed[:, 0])), 10.0, places=4)  # 1 m per 0.1 s
        self.assertEqual(match.provenance["phase_count"], 2)

    def test_undetected_ball_is_missing_and_players_option_masks_extrapolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_match(Path(tmp))
            extrapolated = load_skillcorner(*paths, match_id="SC-7").observations
            detected = load_skillcorner(*paths, match_id="SC-7", players="detected").observations
        self.assertFalse(extrapolated.ball_observed[25])  # source frame 10 has no detected ball
        self.assertTrue(extrapolated.ball_observed[23] and extrapolated.ball_observed[0])
        self.assertTrue(extrapolated.observed[:, 1].all())
        self.assertFalse(detected.observed[np.rint(np.arange(51) * 0.4).astype(int) % 2 == 1, 1].any())

    def test_manifest_dispatch_and_id_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_match(folder)
            manifest = folder / "manifest.toml"
            manifest.write_text('[[matches]]\nid = "SC-7"\nformat = "skillcorner"\nmatch = "m.json"\n'
                                'tracking = "t.jsonl"\nphases = "p.csv"\n', encoding="utf-8")
            (match,) = list(iter_matches(manifest))
            self.assertEqual(match.provenance["format"], "skillcorner")
            with self.assertRaisesRegex(ValueError, "SC-"):
                load_skillcorner(folder / "m.json", folder / "t.jsonl", folder / "p.csv", match_id="SC-8")


if __name__ == "__main__":
    unittest.main()
