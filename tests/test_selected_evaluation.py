"""Deterministic contracts of the selected ensemble evaluation, without real weights."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import evaluate_model as evaluation
from synthetic_inputs import synthetic_atd


class ConstantHead:
    def __init__(self, probability):
        self.probability = probability

    def predict(self, features, periods):
        return np.full(len(periods), self.probability)


def fixture_manifests(root):
    """Five distinct acquisition files, grouped like DFL/SC20/SC9, all invented."""
    paths = []
    for group, ids in (("dfl", ("DFL-A", "DFL-B", "DFL-C")), ("sc", ("SC-1",)), ("legacy", ("SCL-1",))):
        lines = []
        for mid in ids:
            data = {"fps": 25, "pitch": [100, 50], "slot_team": ["H", "A"], "n_frames": 60,
                    "periods": [[1, 2, 22], [2, 24, 60]], "records": [],
                    "reference": list("XX" + "H" * 20 + "XX" + "A" * 20 + "D" * 16),
                    "fixture_identity": mid}
            (root / f"{mid}.json").write_text(json.dumps(data), encoding="utf-8")
            lines.append(f'[[matches]]\nid = "{mid}"\nformat = "synthetic"\npath = "{mid}.json"\n')
        path = root / f"{group}.toml"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append([path])
    return paths


class SelectedEvaluationTests(unittest.TestCase):
    def test_held_out_identity_excludes_all_augmented_copies(self):
        samples = [SimpleNamespace(match_id=mid) for mid in ("DFL-A", "DFL-B", "DFL-A", "SC-1")]
        with patch("train_final.training_inputs", return_value=iter(samples)):
            selected = list(evaluation.training_fold([Path("fixture.toml")], [["combined_dead:0"]], "DFL-A"))
        self.assertEqual([sample.match_id for sample in selected], ["DFL-B", "SC-1"])

    def test_plan_counts_all_four_components_and_never_holds_out_skillcorner(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = fixture_manifests(Path(directory))
            plan = evaluation.audited_plan(*groups)
        self.assertEqual([fold["held_out_match_id"] for fold in plan["folds"]], ["DFL-A", "DFL-B", "DFL-C"])
        self.assertEqual(plan["cost"]["binary_head_fits"], 72)
        components = plan["folds"][0]["components"]
        self.assertEqual([c["training_inputs"] for c in components], [3, 7, 8, 9])
        self.assertEqual(["SCL-1" in c["train_match_ids"] for c in components], [False, False, True, False])
        for fold in plan["folds"]:
            for component in fold["components"]:
                self.assertNotIn(fold["held_out_match_id"], component["train_match_ids"])

    def test_conflicting_acquisitions_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = fixture_manifests(Path(directory))
            with patch("possession.data.external.iter_matches") as loader:
                with self.assertRaisesRegex(ValueError, "duplicate match id"):
                    evaluation.audited_plan(groups[0], groups[0], groups[2])
                loader.assert_not_called()

    def test_invalid_seeds_or_scenarios_never_reach_acquisition_io(self):
        for options in ({"seeds": [1, 1]}, {"seeds": [True]}, {"scenarios": ["clean", "clean"]},
                        {"scenarios": ["tracking_dead"]}):
            with self.subTest(options=options), patch("possession.data.provenance.audit_manifests") as audit:
                with self.assertRaises(ValueError):
                    evaluation.audited_plan([Path("d")], [Path("s")], [Path("l")], **options)
                audit.assert_not_called()

    def test_name_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluation, "ROOT", Path(directory)):
            for name in ("../escape", "nested/run", "nested\\run", "", "..", ".", "/absolute"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evaluation.evaluation_directory(name)
            output = evaluation.evaluation_directory("run-1")
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation.evaluation_directory("run-1")

    def test_tta_is_applied_to_each_member_with_its_recipe(self):
        # Swapped H=.2 maps back to H=.8. D=.2/.8 cancels to .5 in logit space.
        class ViewHead:
            def __init__(self, original, swapped):
                self.original, self.swapped = original, swapped

            def predict(self, features, periods):
                return np.full(len(periods), self.original if features[0, 0] == 1 else self.swapped)

        data = synthetic_atd()
        features = lambda data: (np.ones((data.n_frames, 1)), ["synthetic"])
        swapped = lambda data, recipe: (np.zeros((data.n_frames, 1)), np.zeros((data.n_frames, 1)))
        result = {"learner": "gru", "members": [(1, ViewHead(.2, .8), ViewHead(.8, .2)),
                                                   (2, ViewHead(.8, .2), ViewHead(.2, .8))]}
        with patch.dict("possession.learning.workflow.RECIPES", {"gru11b": (features, features)}), \
                patch("possession.learning.workflow.swapped_features", side_effect=swapped) as swap:
            pairs = evaluation.member_probabilities(result, data, "gru11b")
        np.testing.assert_allclose(pairs[0][0], .8)
        np.testing.assert_allclose(pairs[1][0], .2)
        np.testing.assert_allclose(pairs[0][1], .5)
        np.testing.assert_allclose(pairs[1][1], .5)
        swap.assert_called_once_with(data, "gru11b")

    def test_synthetic_folds_fit_selected_components_and_score_full_periods(self):
        from train_final import training_inputs
        fitted, scored = [], []

        def inputs(*args):
            for sample in training_inputs(*args):
                if sample.match_id == "SCL-1":
                    sample = replace(sample, provenance={**sample.provenance, "dead_supervision": False})
                yield sample

        def fitter(samples, *, recipe, ensemble_seeds, possession_weights):
            samples = list(samples)
            fitted.append((recipe, [s.match_id for s in samples], possession_weights))
            return {"learner": "gru", "test_probability": .8 if len(fitted) % 4 == 1 else .5,
                    "members": [(seed, ConstantHead(.3), ConstantHead(.8))
                                                       for seed in ensemble_seeds]}

        def scorer(reference, labels, periods, fps):
            self.assertEqual(fps, 25)
            self.assertEqual(len(reference), 60)
            self.assertEqual(len(labels), 60)
            np.testing.assert_array_equal(np.flatnonzero(periods == 0), [0, 1, 22, 23])
            self.assertTrue((reference[periods == 0] == "X").all())
            scored.append(True)
            return {"spellF1": .75, "macroF1": .8, "alive_dead": {"f1": .9}}

        with tempfile.TemporaryDirectory() as directory:
            groups = fixture_manifests(Path(directory))
            plan = evaluation.audited_plan(*groups, seeds=[1], scenarios=["clean"])
            config = tomllib_config()
            from possession.learning.decoding import decode_variant
            def probabilities(result, data, recipe):
                return [(np.full(data.n_frames, result["test_probability"]), np.full(data.n_frames, .3))]
            with patch("train_final.training_inputs", side_effect=inputs), \
                    patch.object(evaluation, "member_probabilities", side_effect=probabilities), \
                    patch("possession.learning.decoding.decode_variant", wraps=decode_variant) as decode:
                report = evaluation.evaluate_folds(plan, *groups, config, fitter=fitter, scorer=scorer)
            # Equal weight over the four component members: odds 4,1,1,1 -> odds sqrt(sqrt(4)).
            for call in decode.call_args_list:
                np.testing.assert_allclose(call.args[0], .585786437626905)
                np.testing.assert_allclose(call.args[1], .3)
        self.assertEqual(len(fitted), 12)
        self.assertEqual(len(scored), 3)
        for fold_index, held_out in enumerate(("DFL-A", "DFL-B", "DFL-C")):
            part = fitted[4 * fold_index:4 * (fold_index + 1)]
            self.assertEqual([row[0] for row in part], ["gru11", "gru11", "gru11", "gru11b"])
            self.assertEqual([len(row[1]) for row in part], [3, 7, 8, 9])
            for _, ids, weights in part:
                self.assertNotIn(held_out, ids)
                self.assertTrue(weights)
        self.assertEqual(report["scenario_means"]["clean"]["spellF1"], .75)
        self.assertIn("not an independent test", report["selection_limit"])

    def test_plan_only_audits_but_does_not_load_or_fit_and_has_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = fixture_manifests(root)
            (root / "configs").mkdir()
            (root / "configs/model.toml").write_text((ROOT / "configs/model.toml").read_text(), encoding="utf-8")
            argv = ["--manifest", str(groups[0][0]), "--extra-manifest", str(groups[1][0]),
                    "--legacy-manifest", str(groups[2][0]), "--name", "plan", "--plan-only"]
            with patch.object(evaluation, "ROOT", root), patch("possession.data.external.iter_matches") as loader, \
                    patch("possession.learning.workflow.fit_final") as fitter:
                self.assertEqual(evaluation.main(argv), 0)
                loader.assert_not_called()
                fitter.assert_not_called()
            receipt = json.loads((root / "artifacts/evaluation/plan/manifest.json").read_text())
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["execution"], "PLAN_ONLY")
            self.assertFalse((root / "artifacts/evaluation/plan/metrics.json").exists())

    def test_audit_failure_prevents_training_and_output_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = fixture_manifests(root)
            argv = ["--manifest", str(groups[0][0]), "--extra-manifest", str(groups[1][0]),
                    "--legacy-manifest", str(groups[2][0]), "--name", "rejected"]
            with patch.object(evaluation, "ROOT", root), \
                    patch("possession.data.provenance.audit_manifests", side_effect=ValueError("audit conflict")), \
                    patch("train_final.training_inputs") as inputs, \
                    patch("possession.learning.workflow.fit_final") as fitter:
                self.assertEqual(evaluation.main(argv), 1)
                inputs.assert_not_called()
                fitter.assert_not_called()
            self.assertFalse((root / "artifacts/evaluation/rejected").exists())

    def test_failed_scoring_retains_failure_receipt_without_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = fixture_manifests(root)
            (root / "configs").mkdir()
            (root / "configs/model.toml").write_text((ROOT / "configs/model.toml").read_text(), encoding="utf-8")
            argv = ["--manifest", str(groups[0][0]), "--extra-manifest", str(groups[1][0]),
                    "--legacy-manifest", str(groups[2][0]), "--name", "failure"]
            with patch.object(evaluation, "ROOT", root), patch("official_checkout.checked_official"), \
                    patch.object(evaluation, "evaluate_folds", side_effect=RuntimeError("synthetic scorer failure")):
                self.assertEqual(evaluation.main(argv), 1)
            receipt = json.loads((root / "artifacts/evaluation/failure/manifest.json").read_text())
            self.assertEqual(receipt["status"], "FAIL")
            self.assertEqual(receipt["error"]["message"], "synthetic scorer failure")
            self.assertFalse((root / "artifacts/evaluation/failure/metrics.json").exists())


def tomllib_config():
    import tomllib
    return tomllib.loads((ROOT / "configs/model.toml").read_text())


if __name__ == "__main__":
    unittest.main()
