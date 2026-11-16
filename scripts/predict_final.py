"""Predict the selected B4T model on CPU and validate its official submission CSV.

    python scripts/predict_final.py --model models/casti-ens4-joint-tta-c2 --name b4t --check-repeat

The tactical ATD metadata/raw are resolved by the pinned official games.py from
--data-dir, MTAH_DATA_DIR or .official/data. Feature versions, weight hashes and
library versions must match the card before inference. The two recipes, per-member
team-swap augmentation and logit averaging follow card order; mixdead evidence is
decoded with one joint H/A/D Viterbi without a known-dead override.

Each new artifacts/final/<name>/ records RUNNING before computation. A candidate
CSV becomes submission.csv only after the strict team preflight and unchanged
official validator pass. Failure records its cause and removes temporary outputs.
The manifest records hashes, environment, elapsed time and Python-traced allocations
(excluding native allocations and process RSS). --check-repeat reloads every head and
recomputes features; --expected-sha256 checks a recorded CSV digest before publication.
No ground truth or score is used.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inference_support import (cli_errors, feature_versions, file_manifest, input_file_manifest,
                               library_versions, output_directory, resolve_atd_inputs,
                               validate_atd_alignment, write_json)  # noqa: E402
from possession.data.atd import load_atd  # noqa: E402
from official_checkout import checked_official  # noqa: E402

CARD = "card.json"
LABELS = ("H", "A", "D", "X")


def base_version(version: str) -> str:
    """``2.5.1+cu124`` and ``2.5.1`` share the CPU inference code; local tags are ignored."""
    return str(version).split("+")[0]


def sha256_argument(value: str) -> str:
    """Accept a complete digest (including PowerShell's uppercase form)."""
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise argparse.ArgumentTypeError("SHA256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def load_card(model: Path, trusted_card_sha256: str | None = None, *, card_manifest: dict | None = None) -> dict:
    """Authenticate trusted card bytes before parsing, then validate its contents.

    The default pin is the selected card. An override must come from an
    independently trusted local training run, never from the card itself.
    Hashes identify trusted artifacts; they do not sandbox or inspect pickle.
    """
    from check_repo import FINAL_MODEL_FILES
    from possession.learning.workflow import RECIPES, _costs
    pin = trusted_card_sha256 or FINAL_MODEL_FILES["models/casti-ens4-joint-tta-c2/card.json"]
    card_bytes = (Path(model) / CARD).read_bytes()
    if hashlib.sha256(card_bytes).hexdigest() != pin.lower():
        raise ValueError("model card SHA256 does not match the trusted card pin")
    if card_manifest is not None:
        card_manifest.update(path=str((Path(model) / CARD).resolve()), sha256=hashlib.sha256(card_bytes).hexdigest(),
                             size_bytes=len(card_bytes))
    card = json.loads(card_bytes)
    if card.get("schema_version") != 1 or card.get("recipe") not in RECIPES:
        raise ValueError("model card: unknown schema or recipe")
    if card.get("learner") != RECIPES[card["recipe"]][5]:
        raise ValueError("model card: learner does not match the recipe")
    if (card.get("features") or {}).get("versions") != feature_versions(card["recipe"]):
        raise ValueError("feature versions differ from the model card; retrain with the current feature semantics")
    decoder = card.get("decoder") or {}
    if "control_switch_cost" not in decoder or "dead_switch_cost" not in decoder:
        raise ValueError("model card: decoder costs missing")
    from possession.learning.decoding import VARIANTS, needs_hand_evidence
    if decoder_variant(card) not in VARIANTS:
        raise ValueError("model card: unknown decoder variant")
    from possession.learning.workflow import DECODER_STRUCTURES
    if decoder.get("structure") not in DECODER_STRUCTURES:
        raise ValueError("model card: unknown decoder structure")
    if needs_hand_evidence(decoder_variant(card)):
        from possession.features.control import ControlConfig
        from possession.features.play_state import PlayStateConfig
        hand = decoder.get("hand_evidence")
        if not isinstance(hand, dict) or not isinstance(hand.get("control"), dict) \
                or not isinstance(hand.get("play_state"), dict):
            raise ValueError(f"model card: {decoder_variant(card)} decoding requires hand_evidence control/play_state configs")
        try:
            ControlConfig(**hand["control"])
            PlayStateConfig(**hand["play_state"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"model card: invalid hand_evidence configs: {error}") from error
    _costs(decoder_config(card))
    # The selected card explicitly records both ensemble options.
    for option in ("test_time_augmentation", "possession_weights"):
        if not isinstance(card.get(option), bool):
            raise ValueError(f"model card: {option} must be true or false")
    members = card.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("model card: no ensemble members")
    for recipe in sorted({member_recipe(card, m) for m in members if isinstance(m, dict)} - {card["recipe"]}):
        if recipe not in RECIPES or RECIPES[recipe][5] != card["learner"]:
            raise ValueError(f"model card: member recipe {recipe} unknown or with another learner")
        if (card.get("features_by_recipe", {}).get(recipe) or {}).get("versions") != feature_versions(recipe):
            raise ValueError(f"model card: feature versions of member recipe {recipe} differ; retrain")
    for member in members:
        for head in ("dead", "control"):
            record = (member.get(head) if isinstance(member, dict) else None) or {}
            path = Path(model) / str(record.get("file", ""))
            if not record.get("file") or Path(record["file"]).name != record["file"] or not path.is_file():
                raise ValueError(f"model card: {head} weights missing")
            if file_manifest(path)["sha256"] != record.get("sha256"):
                raise ValueError(f"model card: {head} weights {record['file']} do not match their SHA256")
    mismatched = []
    for name, version in card.get("versions", {}).items():
        if name == "python":
            continue
        installed = importlib.metadata.version(name)
        if base_version(installed) != base_version(version):
            mismatched.append(f"{name} {installed} != {version}")
    if mismatched:
        raise ValueError("installed libraries differ from the model card: " + "; ".join(mismatched))
    return card


def decoder_variant(card: dict) -> str:
    """Explicit decoding variant selected by the card."""
    return card["decoder"].get("variant")


def decoder_config(card: dict) -> dict:
    decoder = card["decoder"]
    config = {"decoder": {"switch_cost": decoder["control_switch_cost"],
                          "structure": decoder.get("structure")},
              "dead_decoder": {"switch_cost": decoder["dead_switch_cost"]}}
    from possession.learning.decoding import needs_hand_evidence
    if needs_hand_evidence(decoder_variant(card)):
        config.update(control=decoder["hand_evidence"]["control"], play_state=decoder["hand_evidence"]["play_state"])
    return config


def load_members(model: Path, card: dict) -> list[tuple]:
    """Fresh CPU copies of every (dead, control) member, in card order."""
    members = []
    for member in card["members"]:
        heads = []
        for head in ("dead", "control"):
            path = Path(model) / member[head]["file"]
            from possession.learning.sequence import SequenceHead
            heads.append(SequenceHead.load(path, device="cpu", trusted_sha256=member[head]["sha256"]))
        members.append(tuple(heads))
    return members


def member_recipe(card: dict, member: dict) -> str:
    """A member's recipe: its own ``recipe`` (merged ensembles of several recipes) or the card's."""
    return member.get("recipe", card["recipe"])


def mixed_recipe_probabilities(members: list[tuple], card: dict, data) -> tuple[np.ndarray, np.ndarray]:
    """Members of several recipes: features once per recipe, each member on its own recipe's
    features, then one mean log-odds over all members in card order (``average_logits``).

    With ``test_time_augmentation`` each member also predicts on its recipe's
    ``swapped_features`` and combines both views with ``swap_average`` before the mean.
    """
    from possession.learning.workflow import RECIPES, average_logits, predict_members, swapped_features
    recipes = [member_recipe(card, member) for member in card["members"]]
    tta = card.get("test_time_augmentation", False)
    features, swapped = {}, {}
    for recipe in dict.fromkeys(recipes):
        dead_features, control_features = RECIPES[recipe][:2]
        dead_x, dead_names = dead_features(data)
        control_x, control_names = control_features(data)
        spec = card["features"] if recipe == card["recipe"] else card["features_by_recipe"][recipe]
        if list(dead_names) != spec["dead"] or list(control_names) != spec["control"]:
            raise ValueError(f"feature names of recipe {recipe} differ from the model card")
        features[recipe] = (dead_x, control_x)
        if tta:
            swapped[recipe] = swapped_features(data, recipe)
    singles = [predict_members([member], card["learner"], *features[recipe], data.period_id, swapped.get(recipe))
               for member, recipe in zip(members, recipes, strict=True)]
    return average_logits([h for h, _ in singles]), average_logits([d for _, d in singles])


def predict_labels(model: Path, card: dict, data, known_dead) -> np.ndarray:
    """Features, both heads and decoding for one input, on one CPU thread."""
    from threadpoolctl import threadpool_limits
    from possession.learning.decoding import decode_variant
    from possession.learning.workflow import RECIPES, predict_members, swapped_features
    dead_features, control_features = RECIPES[card["recipe"]][:2]
    with threadpool_limits(limits=1):
        if card["learner"] == "gru":
            import torch
            torch.set_num_threads(1)
        members = load_members(model, card)
        if {member_recipe(card, member) for member in card["members"]} != {card["recipe"]}:
            probability_h, probability_d = mixed_recipe_probabilities(members, card, data)
            return decode_variant(probability_h, probability_d, data, decoder_config(card), decoder_variant(card),
                                  known_dead)
        dead_x, dead_names = dead_features(data)
        control_x, control_names = control_features(data)
        if list(dead_names) != card["features"]["dead"] or list(control_names) != card["features"]["control"]:
            raise ValueError("feature names differ from the model card; retrain or check out its commit")
        swapped = swapped_features(data, card["recipe"]) if card.get("test_time_augmentation", False) else None
        probability_h, probability_d = predict_members(members, card["learner"], dead_x, control_x,
                                                       data.period_id, swapped)
        return decode_variant(probability_h, probability_d, data, decoder_config(card), decoder_variant(card),
                              known_dead)


def write_csv(labels: np.ndarray, frame: np.ndarray, path: Path) -> str:
    from possession.evaluation.export import write_submission
    rows = ({"frame": int(f), "label": str(label)} for f, label in zip(frame.tolist(), labels.tolist(), strict=True))
    return write_submission(rows, path, int(frame.size))


def validate_submission(path: Path) -> subprocess.CompletedProcess:
    """Strict team preflight plus the unchanged official validator."""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_submission.py"), str(path)],
                          capture_output=True, text=True)


def measured_prediction(model, card, data, known_dead):
    """Time inference and Python-traced allocations without adding a dependency or claiming RSS."""
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        labels = predict_labels(model, card, data, known_dead)
        return labels, {"elapsed_seconds": round(time.perf_counter() - started, 6),
                        "python_tracemalloc_peak_bytes": tracemalloc.get_traced_memory()[1] if own_trace else None,
                        "timing_scope": "inference including tracemalloc profiling overhead",
                        "memory_scope": "Python-traced allocations during features, weight load and decoding; "
                                        "excludes untracked native allocations and process RSS"}
    finally:
        if own_trace:
            tracemalloc.stop()


@cli_errors
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="selected B4T model directory with its card and frozen weights")
    parser.add_argument("--name", required=True, help="new directory under artifacts/final/")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="official data folder; default MTAH_DATA_DIR, else .official/data")
    parser.add_argument("--check-repeat", action="store_true",
                        help="predict twice with fresh model loads and require identical bytes")
    parser.add_argument("--expected-sha256", type=sha256_argument, default=None,
                        help="require this previously recorded CSV SHA256 before publishing the output")
    parser.add_argument("--trusted-card-sha256", type=sha256_argument, default=None,
                        help="independently trusted SHA256 of a locally trained model card")
    args = parser.parse_args(argv)
    model = args.model.resolve()
    authenticated_card = {}
    card = load_card(model, trusted_card_sha256=args.trusted_card_sha256, card_manifest=authenticated_card)
    out = output_directory(ROOT, args.name, "final")
    data_dir = str(args.data_dir.resolve()) if args.data_dir is not None else None
    started = time.perf_counter()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()

    out.mkdir(parents=True, exist_ok=False)
    csv_path, candidate_path, repeat_path = out / "submission.csv", out / "submission.candidate.csv", out / "repeat.csv"
    manifest_path = out / "manifest.json"
    published = False
    recipes = sorted({member_recipe(card, member) for member in card["members"]})
    method = f"learned {card['recipe']} ({card['learner']})"
    if card.get("components"):
        method = (f"ensemble of {len(card['components'])} {card['learner']} models ({', '.join(recipes)}), "
                  f"{decoder_config(card)['decoder']['structure']} decoder")
    manifest = {"schema_version": 1, "status": "RUNNING", "method": method,
                "command": ["scripts/predict_final.py", *(sys.argv[1:] if argv is None else argv)],
                "commit": commit, "working_tree_dirty": bool(dirty), "data_dir": data_dir,
                "model": {"path": str(model), "card": authenticated_card, "name": card.get("name"),
                          "decoder": card["decoder"], "members": card["members"],
                          "test_time_augmentation": card.get("test_time_augmentation", False)},
                "inputs": {},
                "validation": {"tool": "scripts/check_submission.py", "state": "NOT RUN", "exit_code": None},
                "inference": "CPU; threadpool limit 1; torch threads 1 for BiGRU",
                "environment": {"device": "cpu", "native_threads": 1, "system": platform.system(),
                                "machine": platform.machine(), "versions": library_versions(card["learner"])},
                "evaluation": "NOT RUN; no ground truth or score was used"}
    if args.expected_sha256 is not None:
        manifest.update(expected_sha256=args.expected_sha256, expected_sha256_check="NOT RUN")
    try:
        write_json(manifest_path, manifest)
        checked_official()
        data = load_atd("tactical", data_dir=data_dir, use_cache=False)
        input_paths = resolve_atd_inputs(data_dir)
        manifest["inputs"] = {"source": "tactical", "metadata": input_file_manifest(input_paths["metadata"]),
                              "raw": input_file_manifest(input_paths["raw"]), "atd_provenance": data.provenance}
        validate_atd_alignment(data)
        known_dead = None
        manifest["known_dead"] = None
        labels, manifest["inference_measurement"] = measured_prediction(model, card, data, known_dead)
        digest = write_csv(labels, data.frame, candidate_path)
        manifest["candidate_sha256"] = digest
        if args.expected_sha256 is not None:
            matched = digest == args.expected_sha256
            manifest["expected_sha256_check"] = "IDENTICAL" if matched else "DIFFERENT"
            if not matched:
                raise ValueError(f"prediction SHA256 {digest} does not match expected SHA256 {args.expected_sha256}")
        if args.check_repeat:
            repeated_labels, manifest["repeat_inference_measurement"] = measured_prediction(model, card, data, known_dead)
            repeat = write_csv(repeated_labels, data.frame, repeat_path)
            identical = repeat == digest and repeat_path.read_bytes() == candidate_path.read_bytes()
            repeat_path.unlink()
            manifest["repeat_check"] = "IDENTICAL" if identical else "DIFFERENT"
            if not identical:
                raise ValueError("repeat prediction is not byte-identical")
        manifest["validation"].update(state="RUNNING")
        write_json(manifest_path, manifest)
        check = validate_submission(candidate_path)
        manifest["validation"] = {"tool": "scripts/check_submission.py",
                                  "state": "PASS" if check.returncode == 0 else "FAIL", "exit_code": check.returncode}
        if check.returncode:
            print(check.stdout, end="")
            print(check.stderr, end="", file=sys.stderr)
            raise ValueError(f"C4 preflight / official validator failed (exit={check.returncode})")
        shares = {label: round(float(np.mean(labels == label)), 4) for label in LABELS}
        # Publish without overwriting even if another process places a file in this new run.
        if os.name == "nt":
            os.rename(candidate_path, csv_path)
        else:
            os.link(candidate_path, csv_path)
        published = True
        candidate_path.unlink(missing_ok=True)
        manifest.pop("candidate_sha256", None)
        manifest["output"] = {"name": csv_path.name, "sha256": digest}
        manifest.update(status="PASS", label_shares=shares, rows=int(labels.size),
                        elapsed_seconds=round(time.perf_counter() - started, 1))
        write_json(manifest_path, manifest)
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if "output" in manifest:
            manifest["candidate_sha256"] = manifest.pop("output")["sha256"]
        manifest.update(status="FAIL", error=f"{type(error).__name__}: {error}", interrupted=interrupted,
                        elapsed_seconds=round(time.perf_counter() - started, 6), cleanup_errors=[])
        if manifest["validation"]["state"] == "RUNNING":
            manifest["validation"].update(state="FAIL", exit_code=130 if interrupted else None)

        def persist_failure():
            try:
                write_json(manifest_path, manifest)
            except (OSError, TypeError, ValueError) as cleanup_error:
                message = f"{type(cleanup_error).__name__}: {cleanup_error}"
                manifest["cleanup_errors"].append({"name": manifest_path.name, "operation": "write_manifest", "error": message})
                print(f"FAIL write_manifest {manifest_path.name}: {message}", file=sys.stderr)

        # Save the original cause before attempting cleanup, which may hit Windows locks.
        persist_failure()
        for candidate in (candidate_path, repeat_path, *((csv_path,) if published else ())):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as cleanup_error:
                message = f"{type(cleanup_error).__name__}: {cleanup_error}"
                manifest["cleanup_errors"].append({"name": candidate.name, "operation": "unlink", "error": message})
                print(f"FAIL cleanup {candidate.name}: {message}", file=sys.stderr)
        if manifest["cleanup_errors"]:
            persist_failure()
        print(f"FAIL {manifest['error']}", file=sys.stderr)
        return 130 if interrupted else 1
    print(f"rows {labels.size}")
    print("label_shares " + json.dumps(shares))
    print(f"sha256 {digest}")
    print("PASS strict preflight and official validator"
          + ("; repeat byte-identical" if args.check_repeat else "")
          + ("; expected SHA256 matched" if args.expected_sha256 is not None else ""))
    print(f"csv {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
