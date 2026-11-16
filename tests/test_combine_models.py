"""combine_models.py: equal-weight merge of same-recipe models into one card."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "combine_models.py"


def make_model(root, name, payload, decoder=None, members=2):
    folder = root / "models" / name
    folder.mkdir(parents=True)
    card = {"schema_version": 1, "name": name, "recipe": "gru11", "learner": "gru",
            "features": {"versions": {"x": "1"}, "dead": ["a"], "control": ["b"]},
            "decoder": decoder or {"control_switch_cost": 50.0, "dead_switch_cost": 50.0, "variant": "mixdead"},
            "test_time_augmentation": False, "possession_weights": True, "versions": {"torch": "2.5.1"},
            "ensemble": "mean logit", "members": [], "training": {"scenarios": [name]}, "command": [name]}
    for seed in range(members):
        member = {"seed": seed}
        for head in ("dead", "control"):
            path = folder / f"{head}_seed{seed}.pt"
            path.write_bytes(f"{payload}-{head}-{seed}".encode())
            member[head] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_bytes": path.stat().st_size}
        card["members"].append(member)
    (folder / "card.json").write_text(json.dumps(card), encoding="utf-8")


def run(root, *args):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), str(root), *args],
                          capture_output=True, text=True, check=False)


class CombineModelsTests(unittest.TestCase):
    def test_merge_lists_every_member_with_prefixed_rehashed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_model(root, "a", "A")
            make_model(root, "b", "B")
            result = run(root, "--name", "ens", "--decoder-structure", "joint", "models/a", "models/b")
            self.assertEqual(result.returncode, 0, result.stderr)
            card = json.loads((root / "models" / "ens" / "card.json").read_text(encoding="utf-8"))
            self.assertEqual([m["component"] for m in card["members"]], [0, 0, 1, 1])
            self.assertEqual(card["decoder"]["structure"], "joint")
            self.assertTrue(card["test_time_augmentation"])
            self.assertEqual([c["name"] for c in card["components"]], ["a", "b"])
            for member in card["members"]:
                for head in ("dead", "control"):
                    path = root / "models" / "ens" / member[head]["file"]
                    self.assertTrue(member[head]["file"].startswith(f"c{member['component']}_"))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), member[head]["sha256"])

    def test_different_decoders_or_member_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_model(root, "a", "A")
            make_model(root, "b", "B", decoder={"control_switch_cost": 100.0, "dead_switch_cost": 50.0})
            make_model(root, "c", "C", members=3)
            self.assertNotEqual(run(root, "--name", "x", "models/a", "models/b").returncode, 0)
            self.assertNotEqual(run(root, "--name", "y", "models/a", "models/c").returncode, 0)
            self.assertFalse((root / "models" / "x").exists())
            self.assertFalse((root / "models" / "y").exists())

    def test_tampered_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_model(root, "a", "A")
            make_model(root, "b", "B")
            (root / "models" / "b" / "dead_seed0.pt").write_bytes(b"changed")
            self.assertNotEqual(run(root, "--name", "z", "models/a", "models/b").returncode, 0)
            self.assertFalse((root / "models" / "z").exists())

    def test_merged_or_repeated_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_model(root, "a", "A")
            make_model(root, "b", "B")
            self.assertEqual(run(root, "--name", "ab", "models/a", "models/b").returncode, 0)
            make_model(root, "c", "C", members=4)
            nested = run(root, "--name", "n", "models/ab", "models/c")
            self.assertNotEqual(nested.returncode, 0)
            self.assertIn("already a merged model", nested.stderr)
            twice = run(root, "--name", "t", "models/a", "models/a")
            self.assertNotEqual(twice.returncode, 0)
            self.assertIn("listed twice", twice.stderr)
            self.assertFalse((root / "models" / "n").exists())
            self.assertFalse((root / "models" / "t").exists())


if __name__ == "__main__":
    unittest.main()
