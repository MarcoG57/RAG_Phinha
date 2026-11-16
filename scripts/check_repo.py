"""Check the delivery source, local links and exact B4T files without loading weights."""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY = {".git", ".official", ".venv", ".aws", ".ssh", ".codex", ".agents", "data", "artifacts", "models", "logs", "reports", ".cache", "marco_config", "casti_config", "ariel_config"}
FINAL_MODEL_FILES = {
    "models/casti-ens4-joint-tta-c2/c0_control_seed1.pt": "a3805c2855f0aab6368882b169f22e813e842ccbae0eeeed71723c71b2337d35",
    "models/casti-ens4-joint-tta-c2/c0_control_seed2.pt": "b5ff34424327cbc85ef579228e2c63a2432989fdfdc8026241665c54fbc8c966",
    "models/casti-ens4-joint-tta-c2/c0_control_seed20261029.pt": "f93ad0fc44fe2318ff7503fa7f94a1c02d21715d3572842a5b0714b4a971d32b",
    "models/casti-ens4-joint-tta-c2/c0_dead_seed1.pt": "1b682c8461c05f106f8021d341b68f25557412e7170808b28a044660831ea9af",
    "models/casti-ens4-joint-tta-c2/c0_dead_seed2.pt": "0df5ae6e7a716fafd45ee6b24ef7bd7609ae9db7e27fb133511cb845e1bc70d0",
    "models/casti-ens4-joint-tta-c2/c0_dead_seed20261029.pt": "feb026957336e5fdcf4d43a193de41a402432c90ab6fc1615f41c3505a3a482e",
    "models/casti-ens4-joint-tta-c2/c1_control_seed1.pt": "c0635734374f388e4b8c3bb5ac5a01a452143accd4e36325b26f30689e076eaf",
    "models/casti-ens4-joint-tta-c2/c1_control_seed2.pt": "10d65fb8e90b6105a1e99253e409860ebee6b9acb399f69a80d9f2dc9ab6b2d5",
    "models/casti-ens4-joint-tta-c2/c1_control_seed20261029.pt": "4127241034dfb230dd1197d654c92cbad4b4258adeddd9aba3e6a37c5c29da63",
    "models/casti-ens4-joint-tta-c2/c1_dead_seed1.pt": "0cba6f261e88bc5b7b9ed2c9e799823dfe6bbc9c322d9e1d1b4b2453edc3e39e",
    "models/casti-ens4-joint-tta-c2/c1_dead_seed2.pt": "f622a36e6c73fe665727327ce64884f0fedd36cff9bb89165e6c9fb7e663bd45",
    "models/casti-ens4-joint-tta-c2/c1_dead_seed20261029.pt": "8a893bb196172f11c342ba5440809ba37c037adb721fa4cdbebc380312d8cbda",
    "models/casti-ens4-joint-tta-c2/c2_control_seed1.pt": "6999eae7526cc5654c5ce8b2bc6f9389afea9a4cf9f5d244975a7e015b7076d2",
    "models/casti-ens4-joint-tta-c2/c2_control_seed2.pt": "f1e968cddcf45c2001b23d1c3e90b26b56761d761ecedac2ff0de0406edee2b4",
    "models/casti-ens4-joint-tta-c2/c2_control_seed20261029.pt": "a833d148aa2f7b6137e40225f1590fd9b11a0357198a3329853484c15b9fb8b8",
    "models/casti-ens4-joint-tta-c2/c2_dead_seed1.pt": "9efd2b38fc485f56e619df41097e77bc94c5af5b8ba08bb0d57da3cf731bd380",
    "models/casti-ens4-joint-tta-c2/c2_dead_seed2.pt": "780da780af673bda0c3f0541611b5109877e590083af6d19b36bfd17baa1fbe3",
    "models/casti-ens4-joint-tta-c2/c2_dead_seed20261029.pt": "06608c77a63d1e64fcb0879107874799ba639a248669c9def1ade39f8d1076ec",
    "models/casti-ens4-joint-tta-c2/c3_control_seed1.pt": "9de456bee48dfcb0bd932ffff22931be59d48e84b7a01734ed49055ac7f0cc52",
    "models/casti-ens4-joint-tta-c2/c3_control_seed2.pt": "8afd8c524986b8eae04ec5d601530b717b2f5e23d6949f940d68b34844d144f6",
    "models/casti-ens4-joint-tta-c2/c3_control_seed20261029.pt": "0649015c4f0c926adde3dc8c35e1c03f01ca827be54fb2c2bd4a4492e012aaa4",
    "models/casti-ens4-joint-tta-c2/c3_dead_seed1.pt": "83da2f3ebe78fe655e6df8536d67d788ab1c66eb993864038cd6df94df16fe87",
    "models/casti-ens4-joint-tta-c2/c3_dead_seed2.pt": "9a8a469b1302074064634a901e91a83d109aff0012d35c015d95e38842ee1113",
    "models/casti-ens4-joint-tta-c2/c3_dead_seed20261029.pt": "9cb26286156d42f26321de24d6fb1d21a53640a548b6497e01fa99cde96efb37",
    "models/casti-ens4-joint-tta-c2/card.json": "3e568bf94215eac89e6f0f7da1358015a31e7a1f5c74525be117fc6f066dd5a8",
}
BLOCKED_SUFFIXES = {".csv", ".xml", ".mp4", ".mov", ".avi", ".mkv", ".pl3container", ".parquet", ".feather", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth", ".onnx", ".ipynb"}
REQUIRED = (
    "README.md", "AGENTS.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
    "configs/challenge.toml", "configs/model.toml", "configs/skillcorner-2019.example.toml",
    "docs/RAG_Phinha.md", "docs/SOURCES.md", "docs/TRAINING.md", "requirements.lock.txt",
    "scripts/check_repo.py",
    "scripts/check_submission.py",
    "scripts/combine_models.py",
    "scripts/download_dfl.py",
    "scripts/evaluate_model.py",
    "scripts/fetch_skillcorner.py",
    "scripts/inference_support.py",
    "scripts/official_checkout.py",
    "scripts/predict_final.py",
    "scripts/provider_semantics.py",
    "scripts/train_final.py",
    "scripts/training_support.py",
    "src/possession/data/atd.py",
    "src/possession/data/external.py",
    "src/possession/data/provenance.py",
    "src/possession/data/skillcorner.py",
    "src/possession/data/skillcorner_2019.py",
    "src/possession/data/sportec.py",
    "src/possession/evaluation/export.py",
    "src/possession/evaluation/external.py",
    "src/possession/features/control.py",
    "src/possession/features/play_state.py",
    "src/possession/inference/play_state_viterbi.py",
    "src/possession/learning/__init__.py",
    "src/possession/learning/decoding.py",
    "src/possession/learning/features.py",
    "src/possession/learning/sequence.py",
    "src/possession/learning/workflow.py",
) + tuple(FINAL_MODEL_FILES)


def forbidden_path(path: str) -> bool:
    path = path.replace("\\", "/")
    if path in FINAL_MODEL_FILES:
        return False
    path = path.casefold()
    parts = Path(path).parts
    if (parts and parts[0] in LOCAL_ONLY) or any(
        part in {".git", ".venv", "__pycache__"} for part in parts
    ):
        return True
    name = parts[-1] if parts else ""
    return (
        Path(name).suffix in BLOCKED_SUFFIXES
        or name == ".env"
        or path == "agents.override.md"
        or (name.startswith(".env.") and name != ".env.example")
        or (name.endswith(".txt") and not name.startswith("requirements"))
    )


def final_model_errors(root: Path, paths: list[str], *, staged: bool = False) -> list[str]:
    """Hash reviewed working files and optionally index blobs; never deserialize weights."""
    errors = []
    for path in sorted(set(paths) & set(FINAL_MODEL_FILES)):
        if (root / path).is_file():
            with (root / path).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != FINAL_MODEL_FILES[path]:
                errors.append(f"final model file changed: {path}")
        if staged:
            result = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", f":{path}"],
                capture_output=True,
            )
            if result.returncode:
                cause = result.stderr.decode("utf-8", errors="replace").strip()
                errors.append(f"cannot read staged final model file: {path} ({cause})")
            elif hashlib.sha256(result.stdout).hexdigest() != FINAL_MODEL_FILES[path]:
                errors.append(f"staged final model file changed: {path}")
    return errors


def local_link_errors(root: Path, documents: list[Path]) -> list[str]:
    errors = []
    for document in documents:
        # Inline links are the only local-link style used by this template.
        content = document.read_text(encoding="utf-8")
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = target.strip().strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            destination = (document.parent / target.split("#", 1)[0]).resolve()
            try:
                destination.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{document.relative_to(root)}: link outside repository: {target}")
                continue
            if not destination.is_file():
                errors.append(f"{document.relative_to(root)}: missing link: {target}")
    return errors


def tracked_paths(root: Path) -> list[str]:
    command = ["git", "-C", str(root)]
    top = subprocess.run(command + ["rev-parse", "--show-toplevel"], capture_output=True,
                         text=True, encoding="utf-8")
    if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
        raise ValueError("tracked check requires a Git repository rooted at this delivery")
    result = subprocess.run(command + ["ls-files", "-z"], capture_output=True)
    if result.returncode:
        raise ValueError("git ls-files failed")
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def check(root: Path, tracked: bool = False) -> list[str]:
    errors = [f"missing required file: {path}" for path in REQUIRED if not (root / path).is_file()]
    try:
        config = tomllib.loads((root / "configs/challenge.toml").read_text(encoding="utf-8"))
        challenge = config["challenge"]
        if challenge["frames"] != 199319 or challenge["fps"] != 25:
            errors.append("challenge.toml: frame/fps contract changed")
        if challenge["labels"] != ["H", "A", "D", "X"]:
            errors.append("challenge.toml: label contract changed")
        if challenge["rank_metric"] != "spellF1" or challenge["tie_break"] != "macroF1":
            errors.append("challenge.toml: ranking contract changed")
        if config["upstream"]["commit"] != "f8f2629ff1e0985f81d7e59d8413678514ee1530":
            errors.append("challenge.toml: official source pin changed")
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        errors.append(f"metadata: {error}")
    documents = [path for path in root.rglob("*.md")
                 if not any(part in LOCAL_ONLY for part in path.relative_to(root).parts)]
    errors.extend(local_link_errors(root, documents))
    if tracked:
        try:
            paths = tracked_paths(root)
            indexed = set(paths)
            errors.extend(f"required file not tracked: {path}" for path in REQUIRED if path not in indexed)
            errors.extend(f"local-only path tracked: {path}" for path in paths if forbidden_path(path))
            errors.extend(final_model_errors(root, paths, staged=True))
        except (OSError, ValueError) as error:
            errors.append(str(error))
    else:
        existing = [path for path in FINAL_MODEL_FILES if (root / path).is_file()]
        errors.extend(final_model_errors(root, existing))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="check index paths and model bytes in both index and working tree")
    args = parser.parse_args()
    errors = check(ROOT, tracked=args.tracked)
    for error in errors:
        print(f"FAIL {error}")
    if errors:
        return 1
    print("PASS delivery files, B4T hashes and documentation links")
    if not args.tracked:
        print("NOT RUN Git index checks (use --tracked after git init/add)")
    print("NOT RUN inference, official validator, predictive accuracy or rules clearance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
