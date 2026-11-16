"""Re-evaluate selected B4T with external leave-one-DFL-match-out training.

All four components are refitted for every DFL match. SkillCorner 20 trains every
component; SkillCorner 9 trains only alt2, without dead-ball supervision. The
held-out DFL identity and ALL its degraded copies are excluded before fitting.
The scaler belongs to fit_final and therefore sees training features only.

Use --plan-only first: seven DFL folds with three seeds fit 168 binary heads
(4 components x 3 seeds x 2 heads x 7 folds), each with 4000 optimiser steps.
This is a new evaluation entry point, not a bit-exact replay of historical E12.
The development corpus participated in model selection; this is not independent
test evidence. Historical combined_dead/3-second selection has an unresolved
dependency on published challenge baseline statistics. No challenge GT is input.
Only aggregate metrics are written, under a NEW artifacts/evaluation/<name>/.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from training_support import cli_errors, file_manifest, output_directory, write_json

SCENARIOS = ("clean", "combined_dead", "ball_long_gaps_40")
SEEDS = (20261029, 1, 2)
SELECTION_LIMIT = (
    "Development re-evaluation on data used during selection, not an independent test. "
    "Not a bit-exact historical E12 replay. Historical combined_dead/3-second protocol "
    "selection depended on published challenge baseline statistics; admissibility remains unresolved."
)


@dataclass(frozen=True)
class Component:
    name: str
    recipe: str
    dfl_scenarios: tuple[str, ...]
    legacy: bool = False


COPIES = ("combined_dead:0", "combined_dead:1", "ball_long_gaps_40:1")
COMPONENTS = (
    Component("A", "gru11", ("combined_dead:0",)),
    Component("alt1", "gru11", COPIES),
    Component("alt2", "gru11", COPIES, legacy=True),
    Component("alt4", "gru11b", (*COPIES, "dead_ball_missing:1")),
)


def evaluation_directory(name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("--name must be one path component of letters, digits, '.', '_' or '-'")
    return output_directory(ROOT, name, "evaluation")


def selected_config():
    """Use the shared selected-model config, rejecting drift of the fixed decoder."""
    config = tomllib.loads((ROOT / "configs/model.toml").read_text(encoding="utf-8"))
    if (config["decoder"].get("structure") != "joint" or config["decoder"]["switch_cost"] != 50.0
            or config["dead_decoder"]["switch_cost"] != 50.0):
        raise ValueError("B4T evaluation requires configs/model.toml with joint decoder and costs 50/50")
    return config


def training_fold(paths, scenarios, held_out, *, seed=0, controlled_s=3):
    """Preserve source IDs through augmentation; filter before any fitted processing."""
    from train_final import training_inputs
    for sample in training_inputs(paths, scenarios, seed, controlled_s, []):
        if sample.match_id != held_out:
            yield sample


def audited_plan(primary, extra, legacy, *, seeds=SEEDS, scenarios=SCENARIOS) -> dict:
    """Audit acquisitions before dense loading, then describe every grouped fold."""
    from possession.data.provenance import audit_manifests
    from possession.data.external import _read_manifest
    from possession.learning.workflow import recipe_parameters
    seeds, scenarios = list(seeds), list(scenarios)
    if (not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("ensemble seeds must be distinct integers")
    if not scenarios or len(set(scenarios)) != len(scenarios) or not set(scenarios) <= set(SCENARIOS):
        raise ValueError(f"evaluation scenarios must be distinct values from {SCENARIOS}")
    groups = [list(primary), list(extra), list(legacy)]
    if any(not group for group in groups):
        raise ValueError("DFL, SkillCorner 20 and SkillCorner 9 manifests are all required for B4T")
    paths = [path for group in groups for path in group]
    audit = audit_manifests(paths)
    rows = [[row for path in group for row in _read_manifest(path)[2]] for group in groups]
    synthetic = all(row["format"] == "synthetic" for group in rows for row in group)
    if not synthetic:
        for group, expected in zip(rows, ("sportec", "skillcorner", "skillcorner_legacy"), strict=True):
            if any(row["format"] != expected for row in group):
                raise ValueError("B4T needs Sportec DFL, current SkillCorner and legacy SkillCorner in their respective groups")
    ids = [[row["id"] for row in group] for group in rows]
    if len(ids[0]) < 3:
        raise ValueError("DFL LOMO requires at least three distinct primary matches")
    components = [{"name": c.name, "recipe": c.recipe, "dfl_scenarios": list(c.dfl_scenarios),
                   "extra_scenarios": ["combined_dead:0"], "legacy_train_only": c.legacy,
                   "parameters": recipe_parameters(c.recipe), "possession_weights": True}
                  for c in COMPONENTS]
    folds = []
    for held_out in ids[0]:
        fold = {"held_out_match_id": held_out, "components": []}
        for c in COMPONENTS:
            dfl = [mid for mid in ids[0] if mid != held_out]
            train_ids = dfl + ids[1] + (ids[2] if c.legacy else [])
            fold["components"].append({"name": c.name, "train_match_ids": train_ids,
                "training_inputs": len(dfl) * len(c.dfl_scenarios) + len(ids[1]) + (len(ids[2]) if c.legacy else 0)})
        folds.append(fold)
    return {"model": "B4T", "protocol": "DFL leave-one-match-out; all SkillCorner train-only",
            "evidence_kind": "synthetic" if synthetic else "external", "selection_limit": SELECTION_LIMIT,
            "audit": audit, "groups": {"dfl": ids[0], "skillcorner": ids[1], "legacy": ids[2]},
            "components": components, "folds": folds, "ensemble_seeds": seeds,
            "scenarios": scenarios, "degradation_seed": 0, "controlled_s": 3.0,
            "test_time_augmentation": True, "decoder": {"variant": "mixdead", "structure": "joint",
                "control_switch_cost": 50.0, "dead_switch_cost": 50.0},
            "cost": {"component_fits": len(folds) * 4, "binary_head_fits": len(folds) * 4 * len(seeds) * 2,
                     "steps_per_head": 4000, "wall_time": "not estimated; hardware and data dependent",
                     "memory": "one component's training features and models plus one held-out match at a time"}}


def member_probabilities(result, data, recipe):
    """One TTA pair per member, matching mixed_recipe_probabilities in prediction."""
    from possession.learning.workflow import RECIPES, predict_members, swapped_features
    dead_features, control_features = RECIPES[recipe][:2]
    dead_x, _ = dead_features(data)
    control_x, _ = control_features(data)
    swapped = swapped_features(data, recipe)
    return [predict_members([(dead, control)], result["learner"], dead_x, control_x, data.period_id, swapped)
            for _, dead, control in result["members"]]


def evaluate_folds(plan, primary, extra, legacy, config, *, fitter=None, scorer=None) -> dict:
    """Run the planned fits; injected training/scoring boundaries support tiny tests."""
    from train_final import training_inputs
    from possession.evaluation.external import degrade, score_predictions
    from possession.learning.workflow import average_logits, fit_final
    from possession.learning.decoding import decode_variant
    fitter = fit_final if fitter is None else fitter
    scorer = score_predictions if scorer is None else scorer
    rows = []
    for fold in plan["folds"]:
        held_out = fold["held_out_match_id"]
        # Stream clean inputs; keep only this held-out acquisition in memory.
        held = next((sample for sample in training_inputs(primary, ["clean"], 0, 3, [])
                     if sample.match_id == held_out), None)
        if held is None:
            raise ValueError(f"planned held-out match missing from loader: {held_out}")
        outputs = {scenario: [] for scenario in plan["scenarios"]}
        for component, expected in zip(COMPONENTS, fold["components"], strict=True):
            paths = list(primary) + list(extra) + (list(legacy) if component.legacy else [])
            specs = ([list(component.dfl_scenarios)] * len(primary)
                     + [["combined_dead:0"]] * (len(extra) + (len(legacy) if component.legacy else 0)))
            seen_ids, count = set(), 0

            def checked_inputs():
                nonlocal count
                for sample in training_fold(paths, specs, held_out):
                    if sample.match_id not in expected["train_match_ids"]:
                        raise ValueError("loaded training identity is absent from the audited fold")
                    if sample.match_id in plan["groups"]["legacy"] and sample.provenance.get("dead_supervision") is not False:
                        raise ValueError("legacy SkillCorner must exclude dead-ball supervision")
                    seen_ids.add(sample.match_id)
                    count += 1
                    yield sample

            result = fitter(checked_inputs(), recipe=component.recipe,
                            ensemble_seeds=plan["ensemble_seeds"], possession_weights=True)
            if count != expected["training_inputs"] or seen_ids != set(expected["train_match_ids"]):
                raise ValueError("loaded training inputs disagree with the audited fold")
            if len(result["members"]) != len(plan["ensemble_seeds"]):
                raise ValueError("fitted component must preserve the planned member count")
            for scenario in plan["scenarios"]:
                data = held.observations if scenario == "clean" else degrade(held.observations, scenario, 0, held.reference)
                outputs[scenario].extend(member_probabilities(result, data, component.recipe))
            del result
        scores = {}
        for scenario, probabilities in outputs.items():
            data = held.observations if scenario == "clean" else degrade(held.observations, scenario, 0, held.reference)
            ph = average_logits([p[0] for p in probabilities])
            pd = average_logits([p[1] for p in probabilities])
            labels = decode_variant(ph, pd, data, config, "mixdead")
            scores[scenario] = scorer(held.reference, labels, data.period_id, data.fps)
        rows.append({"match_id": held_out, "metrics": scores})
    return {"schema_version": 1, "model": "B4T", "selection_limit": SELECTION_LIMIT,
            "aggregation": "per-match official metrics; scenario means give every DFL match equal weight",
            "matches": rows, "scenario_means": {
                scenario: _metric_means([row["metrics"][scenario] for row in rows]) for scenario in plan["scenarios"]}}


def _metric_means(rows):
    """Average common numeric metric leaves; retain per-match detail in the report."""
    out = {}
    for key in sorted(set.intersection(*(set(row) for row in rows))):
        values = [row[key] for row in rows]
        if all(isinstance(value, dict) for value in values):
            out[key] = _metric_means(values)
        elif all(isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value) for value in values):
            out[key] = float(np.mean(values))
    return out


@cli_errors
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="primary DFL manifest (repeatable)")
    parser.add_argument("--extra-manifest", type=Path, action="append", required=True, help="SkillCorner 20 train-only manifest")
    parser.add_argument("--legacy-manifest", type=Path, action="append", required=True, help="SkillCorner 9 train-only manifest for alt2")
    parser.add_argument("--name", required=True, help="new name for each attempt; output goes to artifacts/evaluation/<name>")
    parser.add_argument("--controlled-s", type=float, choices=(3.0,), default=3.0)
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=list(SCENARIOS), help="scenarios to evaluate")
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=list(SEEDS), help="training seeds for the ensemble")
    parser.add_argument("--plan-only", action="store_true", help="audit and write fold/cost plan; do not load dense matches or train")
    args = parser.parse_args(argv)
    output = evaluation_directory(args.name)
    started = time.perf_counter()
    plan = audited_plan(args.manifest, args.extra_manifest, args.legacy_manifest,
                        seeds=args.ensemble_seeds, scenarios=args.scenario)
    config = selected_config()
    manifest = {"schema_version": 1, "status": "RUNNING", "execution": "PLAN_ONLY" if args.plan_only else "EVALUATION",
                "selection_limit": SELECTION_LIMIT, "effective_config": config,
                "manifests": [{"name": p.name, **{k: v for k, v in file_manifest(p).items() if k != "path"}}
                              for p in args.manifest + args.extra_manifest + args.legacy_manifest],
                "source": {"sha256": file_manifest(Path(__file__))["sha256"]}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    try:
        write_json(output / "plan.json", plan)
        if not args.plan_only:
            from official_checkout import checked_official
            checked_official()  # fail before expensive fitting if the pinned scorer is unavailable
            report = evaluate_folds(plan, args.manifest, args.extra_manifest, args.legacy_manifest, config)
            write_json(output / "metrics.json", report)
        manifest["status"] = "PASS"
    except BaseException as error:
        manifest["status"] = "FAIL"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        write_json(output / "manifest.json", manifest)
    print(f"PASS {manifest['execution']}: {len(plan['folds'])} DFL folds; {plan['cost']['binary_head_fits']} planned head fits; {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
