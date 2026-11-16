"""Synthetic CPU inference, transaction and integrity regressions."""
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from inference_support import file_manifest
from possession.learning import workflow
from possession.learning.sequence import _Network
import predict_final
from synthetic_inputs import synthetic_atd

def quiet(function, *args):
    if function is predict_final.main and args and "--model" in args[0]:
        argv = list(args[0])
        card_path = Path(argv[argv.index("--model") + 1]) / "card.json"
        if card_path.is_file() and "--trusted-card-sha256" not in argv:
            argv.extend(["--trusted-card-sha256", hashlib.sha256(card_path.read_bytes()).hexdigest()])
        args = (argv, *args[1:])
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = function(*args)
    return code, out.getvalue(), err.getvalue()

class FinalPipelineTests(unittest.TestCase):
    def test_card_requires_independent_trust_before_parsing(self):
        self.make_model("untrusted")
        model = self.models_dir / "untrusted"
        with mock.patch.object(predict_final.json, "loads") as parse:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                predict_final.load_card(model)
            parse.assert_not_called()
        digest = hashlib.sha256((model / "card.json").read_bytes()).hexdigest()
        self.assertEqual(predict_final.load_card(model, trusted_card_sha256=digest)["recipe"], "gru11")

    def test_replaced_weights_rejected_after_card_validation(self):
        self.make_model("replaced")
        model = self.models_dir / "replaced"
        digest = hashlib.sha256((model / "card.json").read_bytes()).hexdigest()
        card = predict_final.load_card(model, trusted_card_sha256=digest)
        (model / card["members"][0]["dead"]["file"]).write_bytes(b"replaced")
        with mock.patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                predict_final.load_members(model, card)
            load.assert_not_called()

    def test_manifest_records_authenticated_card_bytes(self):
        self.make_model("receipt-card")
        model = self.models_dir / "receipt-card"
        original = (model / "card.json").read_bytes()
        real = predict_final.load_card
        def replace_after_load(*args, **kwargs):
            card = real(*args, **kwargs)
            (model / "card.json").write_bytes(b"replaced")
            return card
        with mock.patch.object(predict_final, "load_card", side_effect=replace_after_load):
            code, _, err = self.predict(model, "receipt-card")
        self.assertEqual(code, 0, err)
        manifest = json.loads((self.run_dir("receipt-card") / "manifest.json").read_text())
        self.assertEqual(manifest["model"]["card"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(manifest["model"]["card"]["size_bytes"], len(original))
        self.assertEqual(manifest["model"]["card"]["path"], str((model / "card.json").resolve()))

    def test_selected_card_uses_independent_pin(self):
        from check_repo import FINAL_MODEL_FILES
        card_path = ROOT / "models/casti-ens4-joint-tta-c2/card.json"
        self.assertEqual(hashlib.sha256(card_path.read_bytes()).hexdigest(),
                         FINAL_MODEL_FILES["models/casti-ens4-joint-tta-c2/card.json"])
        self.assertEqual(predict_final.load_card(card_path.parent)["recipe"], "gru11")

    def test_invalid_official_checkout_fails_before_reading_or_predicting(self):
        self.make_model("bad-official")
        failures = (
            "official checkout missing or wrong commit; follow README.md",
            "official tracked files modified; restore/reference deliberately",
        )
        for index, message in enumerate(failures):
            with self.subTest(message=message):
                run_name = f"bad-official-{index}"
                with (mock.patch.object(predict_final, "checked_official",
                                        side_effect=ValueError(message)),
                      mock.patch.object(predict_final, "load_atd", side_effect=AssertionError(
                          "reader reached before official checkout validation")) as reader,
                      mock.patch.object(predict_final, "measured_prediction") as prediction):
                    code, _, error = quiet(predict_final.main, [
                        "--model", str(self.models_dir / "bad-official"), "--name",
                        f"{Path(self.runs.name).name}/{run_name}", "--check-repeat",
                    ])
                self.assertEqual(code, 1)
                self.assertIn(message, error)
                reader.assert_not_called()
                prediction.assert_not_called()
                run = self.run_dir(run_name)
                receipt = json.loads((run / "manifest.json").read_text())
                self.assertEqual(receipt["status"], "FAIL")
                self.assertIn(message, receipt["error"])
                self.assertFalse((run / "submission.csv").exists())


    def setUp(self):
            self.models = tempfile.TemporaryDirectory(prefix="test_final_models_")
            self.models_dir = Path(self.models.name)
            final = ROOT / "artifacts" / "final"
            final.mkdir(parents=True, exist_ok=True)
            self.runs = tempfile.TemporaryDirectory(prefix="test_final_pipeline_", dir=final)
            self.inputs = Path(self.runs.name) / "inputs"
            self.inputs.mkdir()
            self.paths = {"metadata": self.inputs / "synthetic_meta.xml", "raw": self.inputs / "synthetic_raw.txt"}
            for path in self.paths.values():
                path.write_text("synthetic input only\n", encoding="utf-8")
            self.data = synthetic_atd()

    def tearDown(self):
            self.runs.cleanup()
            self.models.cleanup()

    def predict(self, model, run, *extra):
            """BAR-ATH boundary replaced by a synthetic ATD; the validator boundary by the strict preflight."""
            validated = []

            def validator(path):
                from check_submission import strict_preflight
                self.assertEqual(path.name, "submission.candidate.csv")
                self.assertFalse((path.parent / "submission.csv").exists())
                running = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual((running["status"], running["validation"]["state"]), ("RUNNING", "RUNNING"))
                strict_preflight(path, self.data.n_frames)
                validated.append(path)
                return subprocess.CompletedProcess([], 0, stdout="", stderr="")

            with (mock.patch.object(predict_final, "load_atd", return_value=self.data) as loader,
                  mock.patch.object(predict_final, "resolve_atd_inputs", return_value=self.paths),
                  mock.patch.object(predict_final, "validate_submission", side_effect=validator)):
                result = quiet(predict_final.main, ["--model", str(model), "--name",
                                                    f"{Path(self.runs.name).name}/{run}", *extra])
            if result[0] == 0:
                self.assertEqual(loader.call_args, mock.call("tactical", data_dir=None, use_cache=False))
                self.assertEqual(len(validated), 1)
            return result

    def run_dir(self, run):
            return Path(self.runs.name) / run

    def test_runner_ignores_configured_known_dead_intervals(self):
        self.make_model("no-mask")
        self.data.dataset_id = "BAR-ATH"
        challenge = {"challenge": {"match": self.data.dataset_id},
                     "known_dead": {"intervals": [[0, 1000000]]}}
        real = predict_final.measured_prediction
        with (mock.patch.object(predict_final, "challenge_config", return_value=challenge, create=True) as config,
              mock.patch.object(predict_final, "measured_prediction", wraps=real) as prediction):
            code, _, err = self.predict(self.models_dir / "no-mask", "no-mask", "--check-repeat")
        self.assertEqual(code, 0, err)
        config.assert_not_called()
        self.assertEqual(len(prediction.call_args_list), 2)
        for call in prediction.call_args_list:
            self.assertIsNone(call.args[3])
        manifest = json.loads((self.run_dir("no-mask") / "manifest.json").read_text())
        self.assertIsNone(manifest["known_dead"])

    def test_success_reports_created_csv_path(self):
            self.make_model("reported-path")
            code, out, err = self.predict(self.models_dir / "reported-path", "reported-path")
            self.assertEqual(code, 0, err)
            reported = next((line.removeprefix("csv ") for line in out.splitlines()
                             if line.startswith("csv ")), None)
            self.assertIsNotNone(reported, out)
            self.assertEqual(Path(reported), (self.run_dir("reported-path") / "submission.csv").resolve())
            self.assertTrue(Path(reported).is_file())

    def test_stale_feature_version_is_rejected_before_data_read(self):
            code, _, err = self.make_model("stale")
            self.assertEqual(code, 0, err)
            model = self.models_dir / "stale"
            card_path = model / "card.json"
            card = json.loads(card_path.read_text(encoding="utf-8"))
            card["features"]["versions"][-1] = "team-shape-obsolete"
            card_path.write_text(json.dumps(card), encoding="utf-8")
            with mock.patch.object(predict_final, "load_atd") as loader:
                code, _, err = quiet(predict_final.main, ["--model", str(model), "--name",
                                    f"{Path(self.runs.name).name}/stale"])
            self.assertEqual(code, 1)
            self.assertIn("feature versions", err)
            loader.assert_not_called()

    def test_running_receipt_precedes_input_computation_and_interrupt_records_failure(self):
            code, _, err = self.make_model("interrupt")
            self.assertEqual(code, 0, err)
            run = self.run_dir("interrupt")

            def interrupt(*args, **kwargs):
                receipt = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual((receipt["status"], receipt["validation"]["state"]), ("RUNNING", "NOT RUN"))
                raise KeyboardInterrupt()

            with mock.patch.object(predict_final, "load_atd", side_effect=interrupt):
                code, _, err = quiet(predict_final.main, ["--model", str(self.models_dir / "interrupt"),
                                    "--name", f"{Path(self.runs.name).name}/interrupt"])
            self.assertEqual(code, 130, err)
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue(manifest["interrupted"])
            self.assertFalse((run / "submission.csv").exists())

    def test_transaction_failures_preserve_cause_and_cleanup_best_effort(self):
            code, _, err = self.make_model("failures")
            self.assertEqual(code, 0, err)
            model = self.models_dir / "failures"
            real_write = predict_final.write_json
            real_unlink = Path.unlink
            for failure in ("validator", "csv", "cleanup", "receipt", "initial-receipt", "validator-receipt"):
                with self.subTest(failure=failure):
                    run = self.run_dir(failure)

                    def validator(path):
                        self.assertFalse((run / "submission.csv").exists())
                        return subprocess.CompletedProcess([], 2, stdout="", stderr="validator refused")

                    def unlink(path, *args, **kwargs):
                        if failure == "cleanup" and path.name == "submission.candidate.csv":
                            raise PermissionError("candidate locked")
                        return real_unlink(path, *args, **kwargs)

                    def receipt(path, manifest):
                        if ((failure == "receipt" and manifest["status"] in ("PASS", "FAIL")) or
                                (failure == "validator-receipt" and manifest["status"] == "FAIL") or
                                (failure == "initial-receipt" and manifest["status"] == "RUNNING")):
                            raise PermissionError("receipt locked")
                        return real_write(path, manifest)

                    with (mock.patch.object(predict_final, "load_atd", return_value=self.data),
                          mock.patch.object(predict_final, "resolve_atd_inputs", return_value=self.paths),
                          mock.patch.object(predict_final, "validate_submission", side_effect=validator if failure in
                                            ("validator", "cleanup", "validator-receipt") else lambda p: subprocess.CompletedProcess([], 0)),
                          mock.patch.object(Path, "unlink", unlink),
                          mock.patch.object(predict_final, "write_json", side_effect=receipt)):
                        if failure == "csv":
                            with mock.patch.object(predict_final, "write_csv", side_effect=PermissionError("csv locked")):
                                result = quiet(predict_final.main, ["--model", str(model), "--name",
                                               f"{Path(self.runs.name).name}/{failure}"])
                        else:
                            result = quiet(predict_final.main, ["--model", str(model), "--name",
                                           f"{Path(self.runs.name).name}/{failure}"])
                    self.assertEqual(result[0], 1, result[2])
                    self.assertFalse((run / "submission.csv").exists())
                    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                    if failure in ("receipt", "validator-receipt"):
                        self.assertEqual(manifest["status"], "RUNNING")
                        self.assertIn("write_manifest", result[2])
                    else:
                        self.assertEqual(manifest["status"], "FAIL")
                    if failure == "cleanup":
                        self.assertIn("validator failed", manifest["error"])
                        self.assertEqual(manifest["cleanup_errors"][0]["operation"], "unlink")
                        self.assertIn("candidate locked", result[2])
                    elif failure == "csv":
                        self.assertIn("csv locked", manifest["error"])
                    elif failure == "initial-receipt":
                        self.assertIn("receipt locked", manifest["error"])
                    elif failure == "validator-receipt":
                        self.assertIn("validator failed", result[2])

    def test_publication_refuses_existing_files_and_does_not_remove_them(self):
            code, _, err = self.make_model("exclusive")
            self.assertEqual(code, 0, err)
            model = self.models_dir / "exclusive"
            run = self.run_dir("exclusive")

            def validator(path):
                # A concurrent file must survive the failed no-overwrite publication.
                (run / "submission.csv").write_bytes(b"prior artifact\n")
                return subprocess.CompletedProcess([], 0)

            with (mock.patch.object(predict_final, "load_atd", return_value=self.data),
                  mock.patch.object(predict_final, "resolve_atd_inputs", return_value=self.paths),
                  mock.patch.object(predict_final, "validate_submission", side_effect=validator)):
                code, _, err = quiet(predict_final.main, ["--model", str(model), "--name",
                                    f"{Path(self.runs.name).name}/exclusive"])
            self.assertEqual(code, 1, err)
            self.assertEqual((run / "submission.csv").read_bytes(), b"prior artifact\n")
            self.assertFalse((run / "submission.candidate.csv").exists())
            before = (run / "manifest.json").read_bytes()
            self.assertEqual(json.loads(before)["status"], "FAIL")
            with mock.patch.object(predict_final, "load_atd") as loader:
                code, _, err = quiet(predict_final.main, ["--model", str(model), "--name",
                                    f"{Path(self.runs.name).name}/exclusive"])
            self.assertEqual(code, 1, err)
            loader.assert_not_called()
            self.assertEqual((run / "manifest.json").read_bytes(), before)
            self.assertEqual((run / "submission.csv").read_bytes(), b"prior artifact\n")

    def test_locked_publication_records_failure_and_cleans_own_candidate(self):
            code, _, err = self.make_model("locked")
            self.assertEqual(code, 0, err)
            operation = "rename" if predict_final.os.name == "nt" else "link"
            real_publish = getattr(predict_final.os, operation)

            def publish(source, target, *args, **kwargs):
                if Path(source).name == "submission.candidate.csv":
                    raise PermissionError("publication locked")
                return real_publish(source, target, *args, **kwargs)

            with (mock.patch.object(predict_final, "load_atd", return_value=self.data),
                  mock.patch.object(predict_final, "resolve_atd_inputs", return_value=self.paths),
                  mock.patch.object(predict_final, "validate_submission", return_value=subprocess.CompletedProcess([], 0)),
                  mock.patch.object(predict_final.os, operation, side_effect=publish)):
                code, _, err = quiet(predict_final.main, ["--model", str(self.models_dir / "locked"), "--name",
                                    f"{Path(self.runs.name).name}/locked"])
            self.assertEqual(code, 1)
            self.assertIn("publication locked", err)
            run = self.run_dir("locked")
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["validation"]["state"]), ("FAIL", "PASS"))
            self.assertIn("candidate_sha256", manifest)
            self.assertFalse((run / "submission.csv").exists())
            self.assertFalse((run / "submission.candidate.csv").exists())

    def assert_prediction(self, output, run):
            lines = output.splitlines()
            self.assertEqual(lines[0], f"rows {self.data.n_frames}")
            shares = json.loads(lines[1].removeprefix("label_shares "))
            self.assertEqual(set(shares), {"H", "A", "D", "X"})
            self.assertAlmostEqual(sum(shares.values()), 1.0, places=3)
            digest = lines[2].removeprefix("sha256 ")
            csv = self.run_dir(run) / "submission.csv"
            self.assertEqual(file_manifest(csv)["sha256"], digest)
            self.assertFalse((self.run_dir(run) / "repeat.csv").exists())
            manifest = json.loads((self.run_dir(run) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["output"]["sha256"]), ("PASS", digest))
            self.assertEqual(manifest["environment"]["device"], "cpu")
            self.assertEqual(manifest["environment"]["native_threads"], 1)
            self.assertEqual(manifest["environment"]["versions"]["numpy"], np.__version__)
            self.assertGreaterEqual(manifest["inference_measurement"]["elapsed_seconds"], 0)
            self.assertGreater(manifest["inference_measurement"]["python_tracemalloc_peak_bytes"], 0)
            labels = csv.read_text(encoding="utf-8").splitlines()[1:]
            # Frames outside periods are X; inside a period the decoder never emits X.
            outside = self.data.period_id == 0
            self.assertTrue(all(row.endswith(",X") for row, out in zip(labels, outside) if out))
            self.assertFalse(any(row.endswith(",X") for row, out in zip(labels, outside) if not out))
            return digest

    def test_expected_sha256_checks_external_repeat_before_publishing(self):
            code, _, err = self.make_model("expected-hash")
            self.assertEqual(code, 0, err)
            model = self.models_dir / "expected-hash"
            code, out, err = self.predict(model, "hash-reference")
            self.assertEqual(code, 0, err)
            expected = self.assert_prediction(out, "hash-reference")
            reference = self.run_dir("hash-reference")
            original = (reference / "submission.csv").read_bytes()
            previous = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("expected_sha256", previous)
            self.assertNotIn("expected_sha256_check", previous)

            code, out, err = self.predict(model, "hash-matched", "--expected-sha256", expected.upper(),
                                          "--check-repeat")
            self.assertEqual(code, 0, err)
            self.assertEqual(self.assert_prediction(out, "hash-matched"), expected)
            matched = self.run_dir("hash-matched")
            manifest = json.loads((matched / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["expected_sha256"], manifest["expected_sha256_check"]),
                             (expected, "IDENTICAL"))
            self.assertEqual((manifest["status"], manifest["repeat_check"], manifest["validation"]["state"]),
                             ("PASS", "IDENTICAL", "PASS"))
            self.assertEqual((matched / "submission.csv").read_bytes(), original)
            self.assertIn("expected SHA256 matched", out)

            wrong = "0" * 64 if expected != "0" * 64 else "1" * 64
            code, out, err = self.predict(model, "hash-mismatched", "--expected-sha256", wrong,
                                          "--check-repeat")
            self.assertEqual(code, 1)
            self.assertIn("does not match expected SHA256", err)
            self.assertNotIn("PASS", out)
            mismatched = self.run_dir("hash-mismatched")
            manifest = json.loads((mismatched / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["expected_sha256_check"]), ("FAIL", "DIFFERENT"))
            self.assertEqual((manifest["expected_sha256"], manifest["candidate_sha256"]), (wrong, expected))
            self.assertNotIn("output", manifest)
            for name in ("submission.csv", "submission.candidate.csv", "repeat.csv"):
                self.assertFalse((mismatched / name).exists(), name)
            self.assertEqual((reference / "submission.csv").read_bytes(), original)

    def test_expected_sha256_rejects_invalid_format_before_model_or_input_reads(self):
            invalid = ("", "f" * 63, "f" * 65, "g" * 64, "0x" + "f" * 62, "f" * 63 + "\n")
            with (mock.patch.object(predict_final, "load_card") as card,
                  mock.patch.object(predict_final, "load_atd") as loader):
                for index, value in enumerate(invalid):
                    run = f"hash-invalid-{index}"
                    argv = ["--model", str(self.models_dir / "not-read"), "--name",
                            f"{Path(self.runs.name).name}/{run}", "--expected-sha256", value]
                    stderr = io.StringIO()
                    with (self.subTest(value=value), redirect_stdout(io.StringIO()), redirect_stderr(stderr),
                          self.assertRaises(SystemExit) as error):
                        predict_final.main(argv)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("64 hexadecimal characters", stderr.getvalue())
                    self.assertFalse(self.run_dir(run).exists())
                card.assert_not_called()
                loader.assert_not_called()

    def test_predict_refuses_incomplete_card_tampered_weights_and_unequal_repeat(self):
            code, _, err = self.make_model("base")
            self.assertEqual(code, 0, err)
            model = self.models_dir / "base"
            card = json.loads((model / "card.json").read_text(encoding="utf-8"))
            broken = dict(card, decoder={"control_switch_cost": 5.0})
            (model / "card.json").write_text(json.dumps(broken), encoding="utf-8")
            with mock.patch.object(predict_final, "load_atd") as loader:
                code, _, err = quiet(predict_final.main, ["--model", str(model), "--name", f"{Path(self.runs.name).name}/x"])
            self.assertEqual(code, 1)
            self.assertIn("decoder costs", err)
            loader.assert_not_called()

            (model / "card.json").write_text(json.dumps(card), encoding="utf-8")
            weights = model / card["members"][-1]["control"]["file"]
            original = weights.read_bytes()
            weights.write_bytes(original + b"\0")
            code, _, err = self.predict(model, "tampered")
            self.assertEqual(code, 1)
            self.assertIn("SHA256", err)
            weights.write_bytes(original)

            real = predict_final.predict_labels
            calls = []

            def drifting(*args):
                labels = real(*args)
                calls.append(1)
                if len(calls) == 2:
                    labels = labels.copy()
                    labels[-1] = "A" if labels[-1] == "H" else "H"
                return labels

            with mock.patch.object(predict_final, "predict_labels", side_effect=drifting):
                code, _, err = self.predict(model, "drift", "--check-repeat")
            self.assertEqual(code, 1)
            self.assertIn("byte-identical", err)
            self.assertFalse((self.run_dir("drift") / "submission.csv").exists())
            manifest = json.loads((self.run_dir("drift") / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["repeat_check"]), ("FAIL", "DIFFERENT"))

    def test_repeat_loads_fresh_heads(self):
        self.make_model("fresh")
        real = predict_final.load_members
        loaded = []
        def load(*args):
            members = real(*args)
            loaded.append(members)
            return members
        with mock.patch.object(predict_final, "load_members", side_effect=load):
            code, _, err = self.predict(self.models_dir / "fresh", "fresh", "--check-repeat")
        self.assertEqual(code, 0, err)
        self.assertEqual(len(loaded), 2)
        self.assertIsNot(loaded[0][0][0], loaded[1][0][0])
        self.assertIsNot(loaded[0][0][1], loaded[1][0][1])

    def test_mixed_recipes_use_each_member_features_and_average_logits(self):
        data = self.data
        recipes = ("gru11", "gru11b")
        card = {"recipe": recipes[0], "members": [{}, {"recipe": recipes[1]}],
                "learner": "gru", "test_time_augmentation": True, "features_by_recipe": {}}
        for recipe in recipes:
            dead, dn = workflow.RECIPES[recipe][0](data)
            control, cn = workflow.RECIPES[recipe][1](data)
            spec = {"dead": list(dn), "control": list(cn)}
            if recipe == card["recipe"]:
                card["features"] = spec
            else:
                card["features_by_recipe"][recipe] = spec
        outputs = [(np.full(data.n_frames, .2), np.full(data.n_frames, .3)),
                   (np.full(data.n_frames, .8), np.full(data.n_frames, .7))]
        members = [(object(), object()), (object(), object())]
        with mock.patch.object(workflow, "predict_members", side_effect=outputs) as predict:
            home, dead = predict_final.mixed_recipe_probabilities(members, card, data)
        np.testing.assert_allclose(home, .5)
        np.testing.assert_allclose(dead, .5)
        self.assertEqual(predict.call_count, 2)
        for call, recipe in zip(predict.call_args_list, recipes, strict=True):
            expected = workflow.RECIPES[recipe][1](data)[0]
            np.testing.assert_array_equal(call.args[3], expected)
            self.assertIsNotNone(call.args[5])

    def test_train_selected_recipe_then_predict_twice_on_synthetic_data(self):
        import train_final
        args = ["--manifest", str(ROOT / "configs/external.synthetic.toml"),
                "--recipe", "gru11", "--control-cost", "50", "--dead-cost", "50",
                "--decoder-structure", "joint", "--decode-variant", "mixdead",
                "--tta", "--possession-weights", "--ensemble-seeds", "7", "8",
                "--model-params", json.dumps({"steps": 4, "hidden": 4, "layers": 1, "window_s": 2.0, "batch": 4}),
                "--name", "trained"]
        with (mock.patch.object(train_final, "MODELS", self.models_dir),
              mock.patch("possession.learning.sequence.SequenceHead.__init__.__defaults__", (None, "cpu"))):
            code, _, err = quiet(train_final.main, args)
        self.assertEqual(code, 0, err)
        model = self.models_dir / "trained"
        card = json.loads((model / "card.json").read_text(encoding="utf-8"))
        self.assertEqual((card["learner"], card["recipe"], card["decoder"]["structure"],
                          card["decoder"]["variant"]), ("gru", "gru11", "joint", "mixdead"))
        self.assertEqual([member["dead"]["file"] for member in card["members"]],
                         ["dead_seed7.pt", "dead_seed8.pt"])
        code, out, err = self.predict(model, "trained-run", "--check-repeat")
        self.assertEqual(code, 0, err)
        self.assert_prediction(out, "trained-run")

    def test_removed_sequence_params_option_is_rejected_before_training(self):
        import train_final
        args = ["--manifest", str(ROOT / "configs/external.synthetic.toml"),
                "--control-cost", "50", "--dead-cost", "50", "--name", "old-option",
                "--sequence-params", "{}"]
        with (mock.patch("possession.learning.workflow.fit_final") as fit,
              mock.patch.object(train_final, "MODELS", self.models_dir),
              redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised):
            train_final.main(args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--sequence-params", errors.getvalue())
        fit.assert_not_called()
        self.assertFalse((self.models_dir / "old-option").exists())

    def make_model(self, name):
        # Construct literal inference weights; no fitting, external data or real weights.
        from inference_support import feature_versions
        from possession.features.control import ControlConfig
        from possession.features.play_state import PlayStateConfig
        from dataclasses import asdict
        model = self.models_dir / name
        model.mkdir()
        recipe = "gru11"
        dead_x, dead_names = workflow.RECIPES[recipe][0](self.data)
        control_x, control_names = workflow.RECIPES[recipe][1](self.data)
        parameters = {"hidden": 4, "layers": 1, "dropout": 0.0, "window_s": 2.0}
        member = {}
        for head, x in (("dead", dead_x), ("control", control_x)):
            torch.manual_seed(11 if head == "dead" else 12)
            network = _Network(2 * x.shape[1], 4, 1, 0.0)
            path = model / (head + ".pt")
            torch.save({"state": network.state_dict(), "centre": np.zeros(x.shape[1]),
                        "scale": np.ones(x.shape[1]), "window": 10, "parameters": parameters,
                        "n_features": x.shape[1]}, path)
            member[head] = {"file": path.name, "sha256": file_manifest(path)["sha256"]}
        card = {"schema_version": 1, "recipe": recipe, "learner": "gru",
                "features": {"dead": dead_names, "control": control_names, "versions": feature_versions(recipe)},
                "decoder": {"control_switch_cost": 50., "dead_switch_cost": 50., "variant": "mixdead",
                            "structure": "joint", "hand_evidence": {"control": asdict(ControlConfig()),
                                                                     "play_state": asdict(PlayStateConfig())}},
                "test_time_augmentation": True, "possession_weights": True, "versions": {}, "members": [member]}
        (model / "card.json").write_text(json.dumps(card), encoding="utf-8")
        return 0, "", ""
