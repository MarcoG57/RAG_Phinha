"""SkillCorner 1ed50b7 reader: possession-only labels (never D) and no dead-head supervision."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from possession.data.external import iter_matches  # noqa: E402
from possession.learning import workflow  # noqa: E402


MANIFEST = ('[[matches]]\nid = "SCL-7"\nformat = "skillcorner_legacy"\n'
            'match = "7/match_data.json"\ntracking = "7/structured_data.json"\n')


def write_legacy(folder: Path, match_id=7, n=400):
    """Tiny legacy match: 10 fps, two periods, a ball and two identified players per team."""
    rng = np.random.default_rng(0)
    players = [{"trackable_object": 100 + k, "team_id": 1 if k < 2 else 2} for k in range(4)]
    (folder / "match_data.json").write_text(json.dumps({
        "id": match_id, "home_team": {"id": 1, "name": "Home"}, "away_team": {"id": 2, "name": "Away"},
        "players": players, "ball": {"trackable_object": 55}, "pitch_length": 105, "pitch_width": 68}),
        encoding="utf-8")
    rows = []
    for f in range(n):
        period = None if f < 20 or 190 <= f < 210 else (1 if f < 190 else 2)
        group = None if period is None or f % 50 < 10 else ("home team" if (f // 50) % 2 else "away team")
        data = [] if f % 37 == 0 else [{"trackable_object": 100 + k, "x": float(rng.normal() * 20),
                                        "y": float(rng.normal() * 10)} for k in range(4)]
        if data and f % 5:
            data.append({"trackable_object": 55, "x": float(rng.normal() * 20), "y": float(rng.normal() * 10)})
        rows.append({"frame": f, "period": period, "time": None, "data": data,
                     "possession": {"trackable_object": None, "group": group}})
    (folder / "structured_data.json").write_text(json.dumps(rows), encoding="utf-8")


class LegacyReaderTests(unittest.TestCase):
    def test_reader_labels_and_dead_head_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "7").mkdir()
            write_legacy(root / "7")
            (root / "legacy.toml").write_text('[[matches]]\nid = "SCL-7"\nformat = "skillcorner_legacy"\n'
                                              'match = "7/match_data.json"\ntracking = "7/structured_data.json"\n',
                                              encoding="utf-8")
            match = next(iter_matches(root / "legacy.toml"))
        self.assertEqual(match.observations.fps, 25)
        self.assertTrue(set(np.unique(match.reference)) <= {"H", "A", "X"})
        self.assertTrue((match.reference == "H").any() and (match.reference == "A").any())
        self.assertTrue((match.reference[match.observations.period_id == 0] == "X").all())
        self.assertIs(match.provenance["dead_supervision"], False)
        from possession.data.provenance import audit_manifests
        with tempfile.TemporaryDirectory() as again:
            root2 = Path(again)
            (root2 / "7").mkdir()
            write_legacy(root2 / "7")
            (root2 / "legacy.toml").write_text(MANIFEST, encoding="utf-8")
            audit = audit_manifests([root2 / "legacy.toml"])
            self.assertEqual(audit["match_count"], 1)
        inputs = workflow.head_inputs([match], "gru11", True)
        self.assertFalse(inputs.masks_d[0].any())
        self.assertTrue(inputs.masks_c[0].any())

    def test_wrong_manifest_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_legacy(root)
            from possession.data.skillcorner_2019 import load_skillcorner_2019
            with self.assertRaisesRegex(ValueError, "SCL-"):
                load_skillcorner_2019(root / "match_data.json", root / "structured_data.json", match_id="SCL-8")


if __name__ == "__main__":
    unittest.main()
