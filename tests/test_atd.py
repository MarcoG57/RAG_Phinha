"""Synthetic contract tests for strict streaming ATD preparation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.data.atd import audit_atd, iter_atd, load_metadata

FIXTURE = ROOT / "tests" / "fixtures" / "data" / "atd_minimal.json"


class AtDTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = self.root / "metadata.xml"
        self.raw = self.root / "tracking.txt"
        teams = self.fixture["teams"]
        players = self.fixture["players"]
        xml = ["<SyntheticEPTS>"]
        for team in teams:
            xml.append(f'<Team id="{team["id"]}"><Name>{team["name"]}</Name></Team>')
        for player in players:
            xml.append(f'<Player id="{player["id"]}" teamId="{player["team_id"]}"/>')
        xml.append("</SyntheticEPTS>")
        self.meta.write_text("\n".join(xml), encoding="utf-8")

    def write_raw(self, rows):
        self.raw.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def settings(self):
        return {
            "dataset_id": "synthetic-match",
            "source": "tactical",
            "fps": 25,
            "expected_frames": 4,
            "pitch_length_m": 105.3,
            "pitch_width_m": 68,
            "periods": [
                {"period_id": "first", "start": 0, "stop": 2},
                {"period_id": "second", "start": 3, "stop": 4},
            ],
        }

    def test_metadata_uses_xml_team_order_and_rejects_unknown_team(self):
        metadata = load_metadata(self.meta)
        self.assertEqual(metadata["team_of_slot"], ["H", "A"])
        self.assertEqual(metadata["team_names"], ["Synthetic H", "Synthetic A"])

        broken = self.root / "unknown.xml"
        broken.write_text(
            '<M><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
            '<Player id="1" teamId="ghost"/></M>', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown teamId"):
            load_metadata(broken)

        duplicate = self.root / "duplicate.xml"
        duplicate.write_text(
            '<M><Team id="h"><Name>Home</Name></Team><Team id="h"><Name>Away</Name></Team>'
            '<Player id="1" teamId="h"/></M>', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            load_metadata(duplicate)

    def test_stream_maps_ids_scales_axes_and_preserves_missing_mask(self):
        self.write_raw(self.fixture["raw_lines"])
        metadata = load_metadata(self.meta)
        frames = list(iter_atd(self.raw, metadata, self.settings()))

        self.assertEqual([frame[0]["frame"] for frame in frames], [0, 1, 2, 3])
        first, tracks = frames[0]
        self.assertEqual(first["period_id"], "first")
        self.assertEqual(first["time_s"], 0)
        self.assertTrue(first["ball_observed"])
        expected = self.fixture["expected"]
        self.assertEqual([first["ball_x_m"], first["ball_y_m"]], expected["first_ball_m"])
        self.assertEqual([track["team"] for track in tracks], ["H", "A"])
        self.assertEqual([track["slot_id"] for track in tracks], [0, 1])
        self.assertTrue(tracks[0]["observed"])
        self.assertEqual([tracks[0]["x_m"], tracks[0]["y_m"]],
                         expected["first_home_player_m"])
        self.assertFalse(tracks[1]["observed"])
        self.assertEqual(tracks[1]["observed"], expected["first_away_observed"])
        self.assertIsNone(tracks[1]["x_m"])
        self.assertIsNone(tracks[1]["y_m"])
        self.assertIsNone(frames[2][0]["period_id"])
        self.assertFalse(frames[2][0]["ball_observed"])
        self.assertIsNone(frames[2][0]["ball_x_m"])
        self.assertTrue(audit_atd(self.raw, expected_frames=4, n_slots=2)["valid"])

    def test_audit_rejects_gap_duplicate_and_non_adjacent_disorder(self):
        for ids in ([1, 3, 3, 4], [1, 3, 1, 4]):
            with self.subTest(ids=ids):
                self.write_raw([f"{frame}:0.5,0.5,0.0;0.5,0.5,0.0:0.5,0.5,0.0"
                                for frame in ids])
                report = audit_atd(self.raw, expected_frames=4, n_slots=2)
                self.assertFalse(report["valid"])
                self.assertEqual(report["n_records"], 4)
                self.assertTrue(report["errors"])
                self.assertNotIn("0.5", " ".join(report["errors"]))

    def test_audit_rejects_malformed_structure_and_infinite_values(self):
        self.write_raw([
            "1:0.5,0.5,0.0;0.5,0.5,0.0:0.5,0.5,0.0",
            "2:broken;0.5,0.5,0.0:0.5,0.5,0.0",
            "3:0.5,0.5,0.0;0.5,0.5,0.0:inf,0.5,0.0",
            "4:0.5,0.5,0.0;0.5,0.5,0.0:0.5,0.5,0.0",
        ])
        report = audit_atd(self.raw, expected_frames=4, n_slots=2)
        self.assertFalse(report["valid"])
        self.assertLessEqual(len(report["errors"]), 20)

    def test_iter_fails_before_yielding_if_later_frame_is_invalid(self):
        self.write_raw(self.fixture["raw_lines"][:3] + [
            "4:0.1,0.2,0.0;0.2,0.3,0.0:0.4,0.5,0.0:extra"
        ])
        rows = iter_atd(self.raw, load_metadata(self.meta), self.settings())
        with self.assertRaisesRegex(ValueError, "invalid ATD"):
            next(rows)

    def test_iter_rejects_overlapping_periods_before_reading(self):
        settings = self.settings()
        settings["periods"] = [
            {"period_id": "first", "start": 0, "stop": 3},
            {"period_id": "second", "start": 2, "stop": 4},
        ]
        rows = iter_atd(self.raw, load_metadata(self.meta), settings)
        with self.assertRaisesRegex(ValueError, "periods"):
            next(rows)


if __name__ == "__main__":
    unittest.main()
