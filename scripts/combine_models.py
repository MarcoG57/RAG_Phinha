"""Merge train_final.py models of one recipe into a single equal-weight ensemble model.

    python scripts/combine_models.py . --name casti-ens-cand-alt1 --decoder-structure joint models/A models/B

``predict_final.py`` averages every card member in logit space, in card order. A merged card that
lists the members of all components therefore predicts with the equal-member ensemble evaluated
offline (each component has the same number of members, so components weigh the same). Learner, decoder, test-time augmentation and library versions must be
identical; feature names and versions must agree within each recipe; weight files are copied with a component prefix and re-hashed. Components may use different
recipes of the same learner: every member then records its ``recipe`` and the
card keeps each other recipe's feature names and versions under ``features_by_recipe``. Each component's
training section, command and commit are kept under ``components``. Already merged models and repeated
models are rejected. ``--test-time-augmentation`` makes every member also predict with teams exchanged.
``--decoder-structure`` sets the
merged card's ``decoder.structure`` (joint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

SAME = ("schema_version", "learner", "decoder", "test_time_augmentation", "possession_weights", "versions",
        "ensemble")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--decoder-structure", choices=("joint",), default="joint")
    parser.add_argument("--test-time-augmentation", action="store_true", default=True,
                        help="predict every member also with teams exchanged (workflow.swap_average)")
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if len(args.models) < 2:
        raise SystemExit("need at least two models")
    root = args.root.resolve()
    out = root / "models" / args.name
    if out.exists():
        raise SystemExit(f"{out} already exists")
    cards = [json.loads((root / m / "card.json").read_text(encoding="utf-8")) for m in args.models]
    for model, card in zip(args.models, cards):  # nested merges would relabel members of mixed recipes
        if card.get("learner") != "gru" or card.get("recipe") not in {"gru11", "gru11b"}:
            raise SystemExit(f"{model}: expected a selected BiGRU recipe")
        if "components" in card or any("recipe" in m or "component" in m for m in card["members"]):
            raise SystemExit(f"{model} is already a merged model; combine the original train_final.py models")
    card_hashes = [sha256(root / m / "card.json") for m in args.models]
    if len({(root / m).resolve() for m in args.models}) != len(args.models) or len(set(card_hashes)) != len(cards):
        raise SystemExit("the same model is listed twice; it would silently get double weight")
    for key in SAME:
        values = {json.dumps(c.get(key), sort_keys=True) for c in cards}
        if len(values) != 1:
            raise SystemExit(f"components differ in {key!r}; only one recipe/decoder/environment can be merged")
    by_recipe = {}
    for card in cards:
        known = by_recipe.setdefault(card["recipe"], card["features"])
        if json.dumps(known, sort_keys=True) != json.dumps(card["features"], sort_keys=True):
            raise SystemExit(f"components of recipe {card['recipe']} differ in their features")
    mixed = len(by_recipe) > 1
    sizes = {len(c["members"]) for c in cards}
    if len(sizes) != 1:
        raise SystemExit("components must have the same number of members so that they weigh the same")
    for model, card in zip(args.models, cards):  # every weight file is checked before anything is written
        for member in card["members"]:
            for head in ("dead", "control"):
                if sha256(root / model / member[head]["file"]) != member[head]["sha256"]:
                    raise SystemExit(f"{model}/{member[head]['file']} does not match its card SHA256")
    out.mkdir(parents=True)
    merged = json.loads(json.dumps({k: cards[0][k] for k in SAME if k in cards[0]}))
    merged.update(recipe=cards[0]["recipe"], features=cards[0]["features"])
    if mixed:
        merged["features_by_recipe"] = {r: f for r, f in by_recipe.items() if r != cards[0]["recipe"]}
    if args.test_time_augmentation:
        merged["test_time_augmentation"] = True
    if args.decoder_structure == "joint":
        merged["decoder"]["structure"] = "joint"
    merged.update({"name": args.name, "evidence_kind": cards[0].get("evidence_kind"),
                   "model_parameters": cards[0].get("model_parameters"),
                   "augment_team_swap": cards[0].get("augment_team_swap"),
                   "train_threads": cards[0].get("train_threads"), "device": cards[0].get("device"),
                   "ensemble": "mean logit over all members of all components in card order "
                               "(workflow.average_logits); equal member counts, so equal component weights",
                   "members": [], "components": []})
    for index, (model, card) in enumerate(zip(args.models, cards)):
        prefix = f"c{index}_"
        for member in card["members"]:
            entry = dict(member)
            for head in ("dead", "control"):
                record = dict(member[head])
                source = root / model / record["file"]
                target = out / f"{prefix}{record['file']}"
                shutil.copyfile(source, target)
                record.update(file=target.name, sha256=sha256(target), size_bytes=target.stat().st_size)
                entry[head] = record
            entry["component"] = index
            if mixed:
                entry["recipe"] = card["recipe"]
            merged["members"].append(entry)
        merged["components"].append({"index": index, "name": card.get("name"), "source": model.as_posix(),
                                     "card_sha256": card_hashes[index],
                                     "commit": card.get("commit"), "working_tree_dirty": card.get("working_tree_dirty"),
                                     "command": card.get("command"), "training": card.get("training"),
                                     "recipe": card["recipe"], "model_parameters": card.get("model_parameters"),
                                     "model_seed": card.get("model_seed"), "ensemble_seeds": card.get("ensemble_seeds"),
                                     "parameter_overrides": card.get("parameter_overrides")})
    (out / "card.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"{out}: {len(merged['members'])} members from {len(cards)} components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
