"""Acquisition audit uses tiny synthetic files, never real match data."""
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession.data.external import iter_matches


def manifest(folder, name, match_id, *, fixture="one.json", provenance=""):
    path = folder / name
    path.write_text(provenance + f'\n[[matches]]\nid = "{match_id}"\nformat = "synthetic"\npath = "{fixture}"\n')
    return path


def fixture(path, x=.2):
    path.write_text(json.dumps({"slot_team": ["H", "A"], "fps": 25, "pitch": [100, 50],
                                "n_frames": 1, "periods": [[1, 0, 1]], "reference": ["H"],
                                "records": [{"frame": 0, "players": [[x, .5], [.8, .5]]}]}))


class ExternalProvenanceTests(unittest.TestCase):
    def audit(self, paths):
        self.assertIsNotNone(importlib.util.find_spec("possession.data.provenance"),
                             "acquisition audit must exist before loading dense matches")
        return importlib.import_module("possession.data.provenance").audit_manifests(paths)

    def test_audit_metadata_fingerprints_and_loader_preserve_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder / "one.json")
            path = manifest(folder, "manifest.toml", "one", provenance=
                            '[provenance]\nsource = "invented"\nversion = "v1"\nlicense = "synthetic"\n')
            report = self.audit([path])
            self.assertEqual(report["counts_by_format"], {"synthetic": 1})
            self.assertEqual(report["match_count"], 1)
            row = report["matches"][0]
            self.assertEqual(row["metadata"], {"source": "invented", "version": "v1", "license": "synthetic"})
            self.assertEqual(row["missing_metadata"], [])
            self.assertEqual(len(row["fingerprints"]["path"]["sha256"]), 64)
            self.assertNotIn(str(folder), json.dumps(report))
            (match,) = iter_matches(path)
            self.assertEqual(match.provenance["kind"], "synthetic")
            self.assertIn("fixture", match.provenance)
            self.assertEqual(match.provenance["manifest_provenance"]["corpus"]["source"], "invented")

    def test_duplicate_ids_are_global_across_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder / "one.json")
            paths = [manifest(folder, name, "one") for name in ("first.toml", "extra.toml")]
            with self.assertRaisesRegex(ValueError, "duplicate.*id"):
                self.audit(paths)

    def test_identical_acquisition_under_another_id_is_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder / "one.json")
            (folder / "copy.json").write_bytes((folder / "one.json").read_bytes())
            paths = [manifest(folder, "first.toml", "one"),
                     manifest(folder, "extra.toml", "two", fixture="copy.json")]
            with self.assertRaisesRegex(ValueError, "duplicate.*acquisition"):
                self.audit(paths)

    def test_old_manifests_record_missing_metadata_and_allow_distinct_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder / "one.json")
            fixture(folder / "two.json", x=.3)
            report = self.audit([manifest(folder, "a.toml", "one"),
                                 manifest(folder, "b.toml", "two", fixture="two.json")])
            self.assertEqual(report["match_count"], 2)
            self.assertEqual(report["matches"][0]["missing_metadata"], ["source", "version", "license"])

    def test_per_match_declarations_override_corpus_without_erasing_either(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder / "one.json")
            path = manifest(folder, "manifest.toml", "one", provenance=
                            '[provenance]\nsource = "corpus"\nsource_version = "v1"\nlicense = "fixture"\n')
            with path.open("a") as handle:
                handle.write('[matches.provenance]\nsource = "match source"\n')
            row = self.audit([path])["matches"][0]
            self.assertEqual(row["metadata"]["source"], "match source")
            self.assertEqual(row["metadata"]["version"], "v1")
            (match,) = iter_matches(path)
            self.assertEqual(match.provenance["manifest_provenance"]["corpus"]["source"], "corpus")
            self.assertEqual(match.provenance["manifest_provenance"]["match"]["source"], "match source")

    def test_sportec_source_id_is_validated_without_parsing_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "metadata.xml").write_text('<Root><MatchInformation><General MatchId="toy"/>'
                                                  '</MatchInformation></Root>')
            (folder / "positions.xml").write_text("invented observations")
            path = folder / "manifest.toml"
            path.write_text('[[matches]]\nid = "alias"\nformat = "sportec"\nmetadata = "metadata.xml"\n'
                            'positions = "positions.xml"\n')
            with self.assertRaisesRegex(ValueError, "source.*id"):
                self.audit([path])
            path.write_text(path.read_text().replace('"alias"', '"toy"'))
            row = self.audit([path])["matches"][0]
            self.assertEqual(row["source_match_id"], "toy")
            self.assertEqual(row["provider"], "sportec")

    def test_skillcorner_source_id_is_validated_before_dense_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "match.json").write_text('{"id": 7}')
            (folder / "track.jsonl").write_text("invented tracking")
            (folder / "phases.csv").write_text("invented phases")
            path = folder / "manifest.toml"
            path.write_text('[[matches]]\nid = "SC-8"\nformat = "skillcorner"\nmatch = "match.json"\n'
                            'tracking = "track.jsonl"\nphases = "phases.csv"\n')
            with self.assertRaisesRegex(ValueError, "source.*id"):
                self.audit([path])
            path.write_text(path.read_text().replace('"SC-8"', '"SC-7"'))
            row = self.audit([path])["matches"][0]
            self.assertEqual(row["source_match_id"], "7")


    def test_skillcorner_formats_share_source_identity_despite_different_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "modern.json").write_text('{"id": 7}')
            (folder / "legacy.json").write_text('{"id": 7, "other": true}')
            (folder / "modern.tracking").write_text("modern observations")
            (folder / "legacy.tracking").write_text("different legacy observations")
            (folder / "phases.csv").write_text("phases")
            path = folder / "manifest.toml"
            path.write_text('[[matches]]\nid = "SC-7"\nformat = "skillcorner"\nmatch = "modern.json"\n'
                            'tracking = "modern.tracking"\nphases = "phases.csv"\n'
                            '[[matches]]\nid = "SCL-7"\nformat = "skillcorner_legacy"\nmatch = "legacy.json"\n'
                            'tracking = "legacy.tracking"\n')
            with self.assertRaisesRegex(ValueError, "duplicate physical match identity"):
                self.audit([path])
            (folder / "legacy.json").write_text('{"id": 8}')
            path.write_text(path.read_text().replace('SCL-7', 'SCL-8'))
            self.assertEqual(self.audit([path])["match_count"], 2)


if __name__ == "__main__":
    unittest.main()
