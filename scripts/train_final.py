"""Train final BiGRU heads on external or synthetic manifests.

Each manifest contributes all matches. Scenarios create degraded training copies;
extra manifests may use their own scenarios. The new model card records input hashes,
parameters, decoder costs, versions and weight hashes. Existing models are not overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from training_support import (cli_errors, degradation_specs, file_manifest, finite_values, json_text,  # noqa: E402
                                manifest_kind)
from inference_support import feature_versions, library_versions  # noqa: E402

MODELS = ROOT / "models"
CARD = "card.json"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def model_directory(name: str) -> Path:
    """A new direct child of models/; refuse an existing one before any data is read."""
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError("--name must be one path component of letters, digits, '.', '_' or '-'")
    directory = MODELS / name
    if directory.exists():
        raise ValueError(f"model directory already exists: {directory}; choose a new --name")
    return directory


def relative_name(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT.resolve()).as_posix() if path.is_relative_to(ROOT.resolve()) else path.name


def portable_command(args) -> list[str]:
    """Canonical parsed CLI: all path options are portable, including repeated/abbreviated ones.

    External manifests keep their basename and content hash in training.manifests;
    another machine supplies its local location. Never copy raw argv into the card.
    """
    command = ["scripts/train_final.py"]
    for option, paths in (("--manifest", args.manifest), ("--extra-manifest", args.extra_manifest)):
        for path in paths:
            command.extend((option, relative_name(path)))
    for option, value in (("--recipe", args.recipe), ("--control-cost", args.control_cost),
                          ("--dead-cost", args.dead_cost), ("--name", args.name),
                          ("--seed", args.seed), ("--controlled-s", args.controlled_s),
                          ("--model-seed", args.model_seed)):
        if value is not None:
            command.extend((option, str(value)))
    command.extend(("--decode-variant", args.decode_variant, "--decoder-structure", args.decoder_structure))
    command.extend(("--train-scenario", *args.train_scenario))
    if args.extra_train_scenario is not None:
        command.extend(("--extra-train-scenario", *args.extra_train_scenario))
    command.extend(option for option, flag in (("--tta", args.tta), ("--possession-weights", args.possession_weights))
                   if flag)
    if args.ensemble_seeds is not None:
        command.extend(("--ensemble-seeds", *map(str, args.ensemble_seeds)))
    if args.model_params is not None:
        command.extend(("--model-params", args.model_params))
    return command


def model_overrides(text: str | None, recipe: str) -> dict | None:
    """Validate CLI JSON at the boundary, using the same recipe contract as tuning/fit_final."""
    if text is None:
        return None
    option = "--model-params"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{option} must be a JSON object: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{option} must be a JSON object")
    from possession.learning.workflow import recipe_parameters
    recipe_parameters(recipe, value)
    return value


def fingerprint(path: Path) -> dict:
    """Content identity without local absolute paths, so a committed card stays portable."""
    record = file_manifest(path)
    return {"name": relative_name(path), "sha256": record["sha256"], "size_bytes": record["size_bytes"]}


def input_record(match) -> dict:
    files = {key: {"name": Path(value["path"]).name, "sha256": value["sha256"], "size_bytes": value["size_bytes"]}
             for key, value in match.provenance.items() if isinstance(value, dict) and "sha256" in value}
    return {"match_id": match.match_id, "n_frames": int(match.observations.n_frames),
            "format": match.provenance.get("format", match.provenance.get("kind")), "files": files}


def manifest_ids(paths: list[Path]) -> str:
    """One evidence kind and unique match ids across every manifest, checked before loading."""
    kinds, ids = set(), []
    for path in paths:
        kinds.add(manifest_kind(path))
        ids += [row["id"] for row in tomllib.loads(Path(path).read_text(encoding="utf-8"))["matches"]]
    if len(kinds) != 1:
        raise ValueError("one evidence kind per model; do not mix fixtures and external matches")
    if len(set(ids)) != len(ids):
        raise ValueError("match ids must be unique across manifests")
    return kinds.pop()


def training_inputs(paths, scenarios, seed, controlled_s, records):
    """Yield one variant at a time; the loader holds one dense match at a time.

    A scenario may carry its own degradation seed as ``SCENARIO:SEED``; otherwise ``seed``.
    ``scenarios`` is one list for every path, or one list per path.
    """
    from possession.data.external import iter_matches
    from possession.evaluation.external import degrade
    from provider_semantics import ALREADY_CONTROLLED, controlled
    per_path = scenarios if scenarios and isinstance(scenarios[0], (list, tuple)) else [scenarios] * len(paths)
    for path, path_scenarios in zip(paths, per_path, strict=True):
        specs = degradation_specs(path_scenarios, seed, label="--train-scenario")
        for match in iter_matches(path):
            reference = match.reference
            if controlled_s is not None and match.provenance.get("format") not in ALREADY_CONTROLLED:
                reference = controlled(np.asarray(reference), match.observations.period_id, controlled_s)
            records.append(input_record(match))
            for scenario, scenario_seed in specs:
                observations = (match.observations if scenario == "clean" else
                                degrade(match.observations, scenario, scenario_seed, reference))
                yield replace(match, reference=reference, observations=observations)


def decoder_section(args) -> dict:
    """Record the selected joint decoder and its hand-evidence configuration."""
    config = tomllib.loads((ROOT / "configs" / "model.toml").read_text(encoding="utf-8"))
    return {"control_switch_cost": args.control_cost, "dead_switch_cost": args.dead_cost,
            "source": "explicit CLI; not chosen by train_final.py", "variant": args.decode_variant,
            "structure": args.decoder_structure,
            "hand_evidence": {"source": "configs/model.toml", "control": config["control"],
                              "play_state": config["play_state"]}}



def save_members(directory: Path, result: dict) -> list[dict]:
    """One file per seed and head, listed in the order the ensemble averages them."""
    members = []
    for seed, *models in result["members"]:
        member = {"seed": seed}
        for head, model in zip(("dead", "control"), models, strict=True):
            path, persistence = directory / f"{head}_seed{seed}.pt", "SequenceHead.save (torch)"
            model.save(path)
            record = file_manifest(path)
            member[head] = {"file": path.name, "persistence": persistence, "sha256": record["sha256"],
                            "size_bytes": record["size_bytes"],
                            "kind": "constant" if isinstance(model, int) else type(model).__name__}
        members.append(member)
    return members


@cli_errors
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="external manifest; every match trains (repeatable)")
    parser.add_argument("--extra-manifest", type=Path, action="append", default=[],
                        help="more training matches, e.g. SkillCorner (repeatable)")
    parser.add_argument("--recipe", choices=("gru11", "gru11b"), default="gru11")
    parser.add_argument("--control-cost", type=float, required=True, help="H/A decoder switch cost, chosen elsewhere")
    parser.add_argument("--dead-cost", type=float, required=True, help="D decoder switch cost, chosen elsewhere")
    parser.add_argument("--controlled-s", type=float, default=None,
                        help="train with the controlled reference of this threshold (s); SkillCorner unchanged")
    parser.add_argument("--train-scenario", nargs="+", default=["clean"], metavar="SCENARIO[:SEED]",
                        help="degradations applied to training inputs; several augment the training set; "
                             "SCENARIO:SEED overrides --seed for that copy")
    parser.add_argument("--extra-train-scenario", nargs="+", default=None, metavar="SCENARIO[:SEED]",
                        help="scenarios of the --extra-manifest matches; default: --train-scenario")
    parser.add_argument("--seed", type=int, default=0, help="degradation seed")
    parser.add_argument("--tta", action="store_true",
                        help="record test-time team swap in the card; predict_final.py applies it")
    parser.add_argument("--possession-weights", action="store_true",
                        help="weigh training frames by clip(8 s / spell duration, 0.25, 4)")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--model-seed", type=int, default=None, help="replaces the recipe's fixed model seed")
    seeds.add_argument("--ensemble-seeds", type=int, nargs="+", default=None,
                       help="one model per seed and head; prediction averages their logits")
    parser.add_argument("--model-params", default=None, help="JSON parameter overrides")
    parser.add_argument("--decoder-structure", choices=("joint",), default="joint")
    parser.add_argument("--decode-variant", choices=("mixdead",), default="mixdead")
    parser.add_argument("--name", required=True, help="new directory under models/")
    args = parser.parse_args(argv)
    finite_values([args.control_cost], label="--control-cost")
    finite_values([args.dead_cost], label="--dead-cost")
    if args.controlled_s is not None:
        finite_values([args.controlled_s], positive=True, label="--controlled-s")
    degradation_specs(args.train_scenario, args.seed, label="--train-scenario")  # distinct known scenarios
    if args.extra_train_scenario is not None:
        degradation_specs(args.extra_train_scenario, args.seed, label="--extra-train-scenario")
    if args.ensemble_seeds is not None and len(set(args.ensemble_seeds)) != len(args.ensemble_seeds):
        raise ValueError("--ensemble-seeds must be distinct")
    overrides = model_overrides(args.model_params, args.recipe)
    directory = model_directory(args.name)
    manifests = [path.resolve() for path in args.manifest + args.extra_manifest]
    kind = manifest_ids(manifests)
    from possession.data.provenance import audit_manifests
    acquisition_audit = audit_manifests(manifests)

    from possession.learning.workflow import fit_final
    started = time.perf_counter()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.strip()
    records = []
    extra_scenarios = args.train_scenario if args.extra_train_scenario is None else args.extra_train_scenario
    scenarios = [args.train_scenario] * len(args.manifest) + [extra_scenarios] * len(args.extra_manifest)
    print(f"Training final {args.recipe} heads: {len(manifests)} manifests x {len(args.train_scenario)} scenarios",
          flush=True)
    result = fit_final(training_inputs(manifests, scenarios, args.seed, args.controlled_s, records),
                       args.recipe, model_seed=args.model_seed, ensemble_seeds=args.ensemble_seeds,
                       parameter_overrides=overrides, possession_weights=args.possession_weights)
    card = {
        "schema_version": 1, "name": args.name, "evidence_kind": kind,
        "recipe": args.recipe, "learner": result["learner"], "model_parameters": result["parameters"],
        "model_seed": args.model_seed, "ensemble_seeds": args.ensemble_seeds, "parameter_overrides": overrides,
        "augment_team_swap": result["augment"], "train_threads": result["threads"],
        "test_time_augmentation": args.tta, "possession_weights": result["possession_weights"],
        "device": getattr(result["members"][0][1], "device", "cpu"),
        "ensemble": "mean logit over members in card order (workflow.average_logits)",
        "decoder": decoder_section(args),
        "training": {"manifests": [fingerprint(path) for path in manifests],
                     "acquisition_audit": acquisition_audit,
                     "reference": "provider" if args.controlled_s is None else f"controlled_{args.controlled_s:g}s",
                     "controlled_s": args.controlled_s, "scenarios": args.train_scenario,
                     **({} if args.extra_train_scenario is None else
                        {"extra_scenarios": args.extra_train_scenario}),
                     "degradation_seed": args.seed, "inputs": records, "variants": result["inputs"],
                     "training_rows": result["training_rows"], "train_classes": result["train_classes"]},
        "features": {"versions": feature_versions(args.recipe), **result["feature_names"]},
        "versions": library_versions(result["learner"]),
        "commit": commit, "working_tree_dirty": bool(dirty),
        "command": portable_command(args),
    }
    MODELS.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.name}-", dir=MODELS))
    try:
        card["members"] = save_members(staging, result)
        card["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        (staging / CARD).write_text(json_text(card), encoding="utf-8")
        if directory.exists():
            raise ValueError(f"model directory appeared during training: {directory}")
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"PASS final {args.recipe} model: {len(records)} matches, {result['inputs']} variants; "
          f"{card['elapsed_seconds']} s; {relative_name(directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
