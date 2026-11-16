"""Focused synthetic regressions for the dense ATD reader's strict parser."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.data.atd import _parse_line, audit_atd, iter_atd, load_metadata, read_atd, read_meta


class DenseATDStrictCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = self.root / "metadata.xml"
        self.raw = self.root / "tracking.txt"
        self.meta.write_text(
            '<M><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
            '<Player id="1" teamId="h"/><Player id="2" teamId="a"/></M>',
            encoding="utf-8",
        )

    def read(self):
        return read_atd(
            self.meta, self.raw, source="tactical", pitch=(105.3, 68.0), fps=25,
            n_frames=1,
        )

    def test_parser_preserves_float64_nan_partial_values_and_float32_limits(self):
        frame_id, players, ball_xy = _parse_line(
            "1:0.12345678912345678,nan,3.4028234663852886e38;"
            "nan,0.9876543219876543,-3.4028234663852886e38:"
            "0.3141592653589793,nan,0.1",
            n_slots=2,
            line_no=7,
        )

        self.assertEqual(frame_id, 1)
        expected_players = np.array(
            [
                [0.12345678912345678, np.nan, 3.4028234663852886e38],
                [np.nan, 0.9876543219876543, -3.4028234663852886e38],
            ],
            dtype=np.float64,
        )
        expected_ball_xy = np.array([0.3141592653589793, np.nan], dtype=np.float64)
        self.assertEqual(players.dtype, np.dtype(np.float64))
        self.assertEqual(ball_xy.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(players, expected_players)
        np.testing.assert_array_equal(ball_xy, expected_ball_xy)

    def test_dense_and_stream_readers_agree_on_synthetic_c1_rows(self):
        fixture = json.loads(
            (ROOT / "tests" / "fixtures" / "data" / "atd_minimal.json").read_text(encoding="utf-8")
        )
        teams = fixture["teams"]
        players = fixture["players"]
        xml = ["<SyntheticEPTS>"]
        for team in teams:
            xml.append(f'<Team id="{team["id"]}"><Name>{team["name"]}</Name></Team>')
        for player in players:
            xml.append(f'<Player id="{player["id"]}" teamId="{player["team_id"]}"/>')
        xml.append("</SyntheticEPTS>")
        self.meta.write_text("\n".join(xml), encoding="utf-8")
        self.raw.write_text("\n".join(fixture["raw_lines"]) + "\n", encoding="utf-8")

        settings = {
            "dataset_id": "synthetic-match",
            "source": "tactical",
            "fps": 25,
            "expected_frames": 4,
            "pitch_length_m": 105.3,
            "pitch_width_m": 68.0,
            "periods": [
                {"period_id": "first", "start": 0, "stop": 2},
                {"period_id": "second", "start": 3, "stop": 4},
            ],
        }
        period_codes = {
            period["period_id"]: number
            for number, period in enumerate(settings["periods"], start=1)
        }
        dense = read_atd(
            self.meta,
            self.raw,
            source="tactical",
            pitch=(settings["pitch_length_m"], settings["pitch_width_m"]),
            fps=settings["fps"],
            n_frames=settings["expected_frames"],
            periods=[
                (period_codes[p["period_id"]], p["start"], p["stop"] - 1)
                for p in settings["periods"]
            ],
            dataset_id=settings["dataset_id"],
        )
        metadata = load_metadata(self.meta)
        self.assertEqual(metadata["team_of_slot"], dense.meta.slot_team.tolist())
        self.assertEqual(metadata["team_names"], [dense.meta.home_name, dense.meta.away_name])
        self.assertEqual(read_meta(self.meta).slot_team.tolist(), metadata["team_of_slot"])
        self.assertTrue(audit_atd(self.raw, expected_frames=4, n_slots=len(players))["valid"])

        streamed = list(iter_atd(self.raw, metadata, settings))
        dense_frames = dense.to_frames_df().to_dict("records")
        dense_tracks = dense.to_tracks_df(observed_only=False).to_dict("records")
        self.assertEqual([row[0]["frame"] for row in streamed], [row["frame"] for row in dense_frames])
        self.assertEqual(len(dense_tracks), sum(len(tracks) for _, tracks in streamed))

        code_periods = {code: period_id for period_id, code in period_codes.items()}
        for (frame, tracks), dense_frame in zip(streamed, dense_frames):
            self.assertEqual(frame["dataset_id"], dense_frame["dataset_id"])
            self.assertEqual(frame["source"], dense_frame["source"])
            self.assertEqual(frame["time_s"], dense_frame["time_s"])
            code = int(dense_frame["period_id"])
            self.assertEqual(frame["period_id"], code_periods.get(code))
            self.assertEqual(frame["ball_observed"], bool(dense_frame["ball_observed"]))
            for key in ("ball_x_m", "ball_y_m"):
                if frame["ball_observed"]:
                    self.assertAlmostEqual(frame[key], dense_frame[key], places=5)
                else:
                    self.assertIsNone(frame[key])
                    self.assertTrue(np.isnan(dense_frame[key]))

        for (frame, tracks), dense_track_start in zip(
            streamed, range(0, len(dense_tracks), len(players))
        ):
            for track, dense_track in zip(
                tracks, dense_tracks[dense_track_start:dense_track_start + len(players)]
            ):
                self.assertEqual(track["frame"], dense_track["frame"])
                self.assertEqual(track["slot_id"] + 1, dense_track["slot_id"])
                self.assertEqual(track["team"], dense_track["team"])
                self.assertEqual(track["observed"], bool(dense_track["observed"]))
                for key in ("x_m", "y_m"):
                    if track["observed"]:
                        self.assertAlmostEqual(track[key], dense_track[key], places=5)
                    else:
                        self.assertIsNone(track[key])
                        self.assertTrue(np.isnan(dense_track[key]))

    def test_dense_reader_rejects_infinite_values(self):
        self.raw.write_text(
            "1:0.1,0.2,inf;0.3,0.4,1:0.5,0.6,0.1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "infinito"):
            self.read()

    def test_dense_and_stream_readers_reject_finite_values_outside_float32_range(self):
        self.raw.write_text(
            "1:1e40,0.2,1;0.3,0.4,1:0.5,0.6,0.1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "float32"):
            self.read()

        settings = {
            "dataset_id": "synthetic-match",
            "source": "tactical",
            "fps": 25,
            "expected_frames": 1,
            "pitch_length_m": 105.3,
            "pitch_width_m": 68.0,
            "periods": [{"period_id": 1, "start": 0, "stop": 1}],
        }
        with self.assertRaisesRegex(ValueError, "float32"):
            list(iter_atd(self.raw, load_metadata(self.meta), settings))

    def test_dense_reader_rejects_signed_frame_ids(self):
        self.raw.write_text(
            "+1:0.1,0.2,0.0;0.3,0.4,1:0.5,0.6,0.1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "frame_id"):
            self.read()


if __name__ == "__main__":
    unittest.main()
