"""Behavioral checks for team collisions, publication paths and documentation links."""
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_repo import FINAL_MODEL_FILES, forbidden_path, local_link_errors


class PublicationPathTests(unittest.TestCase):
    def test_required_source_missing_from_disk_and_index_is_reported(self):
        import check_repo
        from unittest import mock

        path = "src/possession/learning/features.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True,
                           capture_output=True)
            module = root / path
            module.parent.mkdir(parents=True)
            module.write_text("# synthetic module\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                           capture_output=True)
            with mock.patch.object(check_repo, "FINAL_MODEL_FILES", {}):
                messages = (f"missing required file: {path}", f"required file not tracked: {path}")
                self.assertTrue(all(message not in check_repo.check(root, tracked=True)
                                    for message in messages))
                subprocess.run(["git", "-C", str(root), "rm", "-f", "--quiet", "--", path],
                               check=True, capture_output=True)
                errors = check_repo.check(root, tracked=True)
                for message in messages:
                    self.assertIn(message, errors)
                module.parent.mkdir(parents=True, exist_ok=True)
                module.write_text("# synthetic module\n", encoding="utf-8")
                subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                               capture_output=True)
                self.assertTrue(all(message not in check_repo.check(root, tracked=True)
                                    for message in messages))

    def test_required_file_must_be_in_git_index_even_if_present_on_disk(self):
        import check_repo
        from unittest import mock

        path = "required.txt"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True,
                           capture_output=True)
            (root / path).write_text("synthetic", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                           capture_output=True)
            with (mock.patch.object(check_repo, "REQUIRED", (path,)),
                  mock.patch.object(check_repo, "FINAL_MODEL_FILES", {})):
                message = f"required file not tracked: {path}"
                self.assertNotIn(message, check_repo.check(root, tracked=True))
                subprocess.run(["git", "-C", str(root), "rm", "--cached", "--quiet", "--", path],
                               check=True, capture_output=True)
                self.assertTrue((root / path).is_file())
                self.assertIn(message, check_repo.check(root, tracked=True))
                subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                               capture_output=True)
                self.assertNotIn(message, check_repo.check(root, tracked=True))

    def test_tracked_models_check_the_index_even_when_working_bytes_are_restored(self):
        import hashlib
        import check_repo
        from unittest import mock

        path = "models/casti-ens4-joint-tta-c2/card.json"
        reviewed = b"reviewed synthetic card\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True,
                           capture_output=True)
            subprocess.run(["git", "-C", str(root), "config", "core.autocrlf", "false"],
                           check=True, capture_output=True)
            model = root / path
            model.parent.mkdir(parents=True)
            model.write_bytes(reviewed)
            subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                           capture_output=True)
            pins = {path: hashlib.sha256(reviewed).hexdigest()}
            with mock.patch.object(check_repo, "FINAL_MODEL_FILES", pins):
                # Missing template metadata is unrelated to this publication boundary.
                self.assertFalse(any("final model" in error
                                     for error in check_repo.check(root, tracked=True)))
                model.write_bytes(b"unreviewed staged card\n")
                subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                               capture_output=True)
                model.write_bytes(reviewed)
                self.assertIn(f"staged final model file changed: {path}",
                              check_repo.check(root, tracked=True))
                subprocess.run(["git", "-C", str(root), "add", "--", path], check=True,
                               capture_output=True)
                model.write_bytes(b"unreviewed working card\n")
                errors = check_repo.check(root, tracked=True)
                self.assertIn(f"final model file changed: {path}", errors)
                self.assertNotIn(f"staged final model file changed: {path}", errors)

    def test_tracked_candidate_bytes_must_match_the_reviewed_pins(self):
        import hashlib
        import check_repo
        from unittest import mock

        path = "models/casti-ens4-joint-tta-c2/card.json"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / path).parent.mkdir(parents=True)
            (root / path).write_bytes(b"synthetic card\n")
            pins = {path: hashlib.sha256(b"reviewed card\n").hexdigest()}
            with (mock.patch.object(check_repo, "FINAL_MODEL_FILES", pins),
                  mock.patch.object(check_repo, "tracked_paths", return_value=[path])):
                self.assertIn(f"final model file changed: {path}", check_repo.check(root, tracked=True))
                (root / path).write_bytes(b"reviewed card\n")
                self.assertNotIn(f"final model file changed: {path}", check_repo.check(root, tracked=True))

    def test_missing_selected_weight_is_reported(self):
        import check_repo
        from unittest import mock
        path = "models/casti-ens4-joint-tta-c2/control_seed1.pt"
        with tempfile.TemporaryDirectory() as directory:
            with (mock.patch.object(check_repo, "FINAL_MODEL_FILES", {path: "0" * 64}),
                  mock.patch.object(check_repo, "REQUIRED", (path,))):
                errors = check_repo.check(Path(directory))
                self.assertTrue(any(path in error for error in errors), errors)

    def test_only_pinned_b4t_files_are_allowed(self):
        self.assertEqual(len(FINAL_MODEL_FILES), 25)
        for path in FINAL_MODEL_FILES:
            with self.subTest(path=path):
                self.assertFalse(forbidden_path(path))
                self.assertFalse(forbidden_path(path.replace("/", "\\")))
        for path in ("models/casti-ens4-joint-tta-c2/private.pt",
                     "models/casti-ens4-joint-tta-c2/nested/card.json",
                     "models/casti-bigru11-ens3sc-c3-cd/card.json",
                     "models/casti-ens4-joint-tta-c2/../card.json"):
            with self.subTest(path=path):
                self.assertTrue(forbidden_path(path))

    def test_gitignore_exposes_only_pinned_files(self):
        allowed = list(FINAL_MODEL_FILES)
        blocked = ["models/casti-ens4-joint-tta-c2/private.pt",
                   "models/casti-ens4-joint-tta-c2/nested/card.json",
                   "models/casti-bigru11-ens3sc-c3-cd/card.json"]
        result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"],
                                cwd=Path(__file__).resolve().parents[1],
                                input=("\0".join(allowed + blocked) + "\0").encode(),
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.decode().rstrip("\0").split("\0")), set(blocked))

    def test_windows_case_variants_cannot_bypass_local_only_paths(self):
        for path in ("DATA/sample.json", "Artifacts/report.md", ".OFFICIAL/source.py",
                     "src/__PYCACHE__/cache.py", ".ENV", ".Env.Local",
                     "AGENTS.OVERRIDE.MD", "private/notes.TXT", ".AWS/credentials", ".SSH/key",
                     ".CODEX/config.toml", ".AGENTS/settings.json", "Marco_config/profile.md"):
            with self.subTest(path=path):
                self.assertTrue(forbidden_path(path))

    def test_restricted_and_generated_paths_are_rejected(self):
        for path in ("data/sample.json", "artifacts/run/report.md", "video.mp4",
                     "notebooks/output.ipynb", ".env", ".env.local", "src/export.csv",
                     ".official/starter/baseline.py", "private/tracking.xml"):
            with self.subTest(path=path):
                self.assertTrue(forbidden_path(path))

    def test_source_and_synthetic_json_can_be_reviewed(self):
        for path in ("src/possession/data/loader.py", "tests/fixtures/synthetic.json",
                     "docs/PLAN.md", "requirements.lock.txt", ".env.example"):
            with self.subTest(path=path):
                self.assertFalse(forbidden_path(path))


class LinkTests(unittest.TestCase):
    def test_existing_link_passes_and_missing_link_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "README.md"
            (root / "target.md").write_text("synthetic", encoding="utf-8")
            document.write_text("[ok](target.md) [web](https://example.com)", encoding="utf-8")
            self.assertEqual(local_link_errors(root, [document]), [])
            document.write_text("[bad](missing.md)", encoding="utf-8")
            self.assertEqual(len(local_link_errors(root, [document])), 1)

    def test_link_outside_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "README.md"
            document.write_text("[outside](../outside.md)", encoding="utf-8")
            self.assertTrue(local_link_errors(root, [document]))


if __name__ == "__main__":
    unittest.main()
