"""Leave-one-match-out learning; references enter only loss and optional scoring.

Two binary heads: H/A on known alive frames, D/alive on known H/A/D frames.
All variants sharing match_id are held out together. No test-based selection,
random frame validation, interpolation or fitted preprocessing is performed.
``fit_final`` fits the same heads once on every input, for a final model.
Options, all off by default: test-time team swap (``swap_average``), possession-duration
loss weights (``spell_weights``) and degraded copies that only train (``train_variants``).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Protocol

import numpy as np
from threadpoolctl import threadpool_limits

from possession.data.atd import AtdData
from possession.inference.play_state_viterbi import decode_joint
from possession.learning.sequence import SEQUENCE_PARAMETERS, SEQUENCE_PARAMETERS_AUDIT
from possession.learning.features import (onpitch_control_features, onpitch_control_features_speed5,
                                          onpitch_player_features, swap_teams)

# Fixed preparation hypothesis, never selected against public/outer-fold GT.
# recipe -> (dead features, control features, model parameters, H/A swap augmentation, threads, learner)
RECIPES = {
    "gru11": (onpitch_player_features, onpitch_control_features, SEQUENCE_PARAMETERS, True, 4, "gru"),
    "gru11b": (onpitch_player_features, onpitch_control_features_speed5, SEQUENCE_PARAMETERS_AUDIT, True, 4, "gru"),
}
METHODS = ("learned_HA_D",)
# Possession-duration weights (idea from the FCB-Hacks entry), fixed a priori: a frame of
# a spell of ``SPELL_WEIGHT_S`` seconds weighs 1, shorter spells more, longer ones less.
SPELL_WEIGHT_S = 8.0
SPELL_WEIGHT_CLIP = (.25, 4.0)


class MatchInput(Protocol):
    match_id: str
    observations: AtdData
    reference: np.ndarray
    provenance: dict


def _costs(config: Mapping) -> tuple[float, float]:
    costs = (config["decoder"]["switch_cost"], config["dead_decoder"]["switch_cost"])
    if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0 for value in costs):
        raise ValueError("decoder costs must be finite non-negative numbers")
    return tuple(float(value) for value in costs)


DECODER_STRUCTURES = ("joint",)


def decoder_structure(config: Mapping) -> str:
    """Require the explicit joint structure selected for B4T."""
    structure = config["decoder"].get("structure")
    if structure not in DECODER_STRUCTURES:
        raise ValueError(f"decoder structure must be one of {DECODER_STRUCTURES}")
    return structure


def _evidence(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite vectors in [0,1]")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # Existing decoder maximizes +/- evidence: half log odds equals the
    # difference between two log emission scores, up to a shared constant.
    return 0.5 * (np.log(p) - np.log1p(-p))


def decode_evidence(control_evidence: np.ndarray, dead_evidence: np.ndarray,
                    period_id: np.ndarray, config: Mapping,
                    known_dead: np.ndarray | None = None) -> np.ndarray:
    """One H/A/D Viterbi with the selected costs and known-dead mask."""
    control_cost, dead_cost = _costs(config)
    decoder_structure(config)
    return decode_joint(control_evidence, dead_evidence, period_id, control_cost, dead_cost, known_dead)


def decode_probabilities(control_probability: np.ndarray, dead_probability: np.ndarray,
                         period_id: np.ndarray, config: Mapping,
                         known_dead: np.ndarray | None = None) -> np.ndarray:
    """``known_dead`` (official alignment, never labels) forces D before control is decoded."""
    _costs(config)
    return decode_evidence(_evidence(control_probability), _evidence(dead_probability), period_id, config,
                           known_dead)


def _validate(sample: MatchInput) -> None:
    data = sample.observations
    n, slots = data.n_frames, data.meta.n_slots
    if not isinstance(sample.match_id, str) or not sample.match_id.strip():
        raise ValueError("match_id must be a nonempty string shared by all its variants")
    if n == 0 or not np.array_equal(data.frame, np.arange(n)):
        raise ValueError("frame alignment must be contiguous from zero")
    if np.asarray(sample.reference).shape != (n,) or not np.isin(sample.reference, ["H", "A", "D", "X"]).all():
        raise ValueError("reference must align one H/A/D/X label per frame")
    shapes = {"period_id": (n,), "record_present": (n,), "ball_xy": (n, 2),
              "ball_observed": (n,), "xy": (n, slots, 2), "speed": (n, slots),
              "observed": (n, slots)}
    for name, shape in shapes.items():
        if np.asarray(getattr(data, name)).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if not np.isin(data.meta.slot_team, ["H", "A"]).all():
        raise ValueError("slot teams must be H/A")
    if not np.issubdtype(data.period_id.dtype, np.integer) or not np.isin(data.period_id, [0, 1, 2, 3, 4]).all():
        raise ValueError("period_id must use dense integer values 0..4")
    for period in (1, 2, 3, 4):
        indices = np.flatnonzero(data.period_id == period)
        if indices.size and not (np.diff(indices) == 1).all():
            raise ValueError("each period must be contiguous")
    if isinstance(data.fps, bool) or not np.isfinite(data.fps) or data.fps <= 0:
        raise ValueError("fps must be positive")
    if len(data.pitch) != 2 or not np.isfinite(data.pitch).all() or min(data.pitch) <= 0:
        raise ValueError("pitch must contain two positive dimensions in metres")


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def average_logits(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Ensemble of aligned probability vectors: mean log-odds, back through the sigmoid.

    One vector is returned unchanged, so a one-member ensemble is the model itself.
    Probabilities are clipped to [1e-6, 1-1e-6] first, as in the decoder evidence.
    """
    arrays = [np.asarray(p, dtype=np.float64) for p in probabilities]
    if not arrays or any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("ensemble members must be nonempty aligned probability vectors")
    if len(arrays) == 1:
        return arrays[0]
    return 1 / (1 + np.exp(-np.mean([_logit(a) for a in arrays], axis=0)))


def swap_average(probability_h: np.ndarray, probability_d: np.ndarray, swapped_h: np.ndarray,
                 swapped_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Test-time team swap: (p_H, p_D) averaged with the prediction on ``swap_teams`` input.

    ``swapped_h`` is the probability that the exchanged home team (the original away
    team) controls, so logit p_H = (logit p_H + logit(1 - p_H')) / 2 and
    logit p_D = (logit p_D + logit p_D') / 2, with ``average_logits`` clipping.
    """
    return (average_logits([probability_h, 1 - np.asarray(swapped_h, dtype=np.float64)]),
            average_logits([probability_d, swapped_d]))


def swapped_features(observations: AtdData, recipe: str,
                     control_x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(dead, control) features of ``recipe`` on teams exchanged and the pitch rotated.

    ``control_x`` reuses swapped control features already computed for augmentation.
    """
    dead_features, control_features_fn = RECIPES[recipe][:2]
    flipped = swap_teams(observations)
    return (dead_features(flipped)[0],
            control_features_fn(flipped)[0] if control_x is None else control_x)


def spell_weights(reference: np.ndarray, period_id: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame loss weight clip(8 s / spell duration, 0.25, 4), float32.

    A spell is a maximal run of one team (H or A) inside a period. X frames are
    skipped: they neither end nor lengthen a spell; D ends it. The duration counts the
    spell's H/A frames. D, X and period-zero frames weigh 1. Only the training
    reference of the same input enters, never another match.
    """
    reference, period_id = np.asarray(reference), np.asarray(period_id)
    weights = np.ones(reference.shape[0], dtype=np.float32)
    low, high = SPELL_WEIGHT_CLIP
    for period in np.unique(period_id[period_id > 0]):
        rows = np.flatnonzero((period_id == period) & (reference != "X"))
        if not rows.size:
            continue
        labels = reference[rows]
        starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        lengths = np.diff(np.r_[starts, labels.size])
        spell = np.repeat(np.clip(SPELL_WEIGHT_S * fps / lengths, low, high), lengths)
        team = np.isin(labels, ["H", "A"])
        weights[rows[team]] = spell[team]
    return weights


def _seed_key(learner: str) -> str:
    return "seed"


def recipe_parameters(recipe: str, overrides: Mapping | None = None) -> dict:
    """Model parameters of ``recipe`` updated with validated ``overrides``.

    Only numeric keys of the recipe's parameter dict may change, to finite positive
    numbers (integers where the recipe uses integers; ``dropout`` below 1). The seed
    stays with ``model_seed``/``ensemble_seeds``.
    """
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    parameters, learner = RECIPES[recipe][2], RECIPES[recipe][5]
    checked = {}
    for key, value in dict(overrides or {}).items():
        if key not in parameters:
            raise ValueError(f"unknown parameter for recipe {recipe}: {key}")
        default = parameters[key]
        if key == _seed_key(learner):
            raise ValueError("set the model seed with model_seed or ensemble_seeds, not overrides")
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            raise ValueError(f"parameter {key} is not numeric and cannot be overridden")
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"parameter {key} must be a finite positive number")
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError(f"parameter {key} must be an integer")
        if key == "dropout" and value >= 1:
            raise ValueError("parameter dropout must be below 1")
        checked[key] = int(value) if isinstance(default, int) else float(value)
    return {**parameters, **checked}


def validate_inputs(matches: Sequence[MatchInput]) -> None:
    for sample in matches:
        _validate(sample)


@dataclass(frozen=True)
class HeadInputs:
    """Per-input features and loss masks, computed once and shared by every fold."""
    match_ids: tuple[str, ...]
    references: tuple[np.ndarray, ...]
    dead_x: tuple[np.ndarray, ...]
    control_x: tuple[np.ndarray, ...]
    swapped_x: tuple[np.ndarray, ...] | None
    masks_d: tuple[np.ndarray, ...]
    masks_c: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...] | None = None  # per-frame loss weights (spell_weights)


def _concatenate(parts: Sequence[HeadInputs]) -> HeadInputs:
    return HeadInputs(**{field.name: None if getattr(parts[0], field.name) is None else
                         tuple(value for part in parts for value in getattr(part, field.name))
                         for field in fields(HeadInputs)})


def head_inputs(matches: Sequence[MatchInput], recipe: str, possession_weights: bool = False) -> HeadInputs:
    """Features of ``recipe`` for inputs already checked by ``validate_inputs``.

    ``possession_weights`` adds ``spell_weights`` of each input's reference.
    """
    dead_features, control_features_fn, _, augment, _, _ = RECIPES[recipe]
    # Compact matrices only, no duplicate tracking tensors. Missing-value encoding
    # and fitted normalisation belong to each training fold.
    return HeadInputs(
        match_ids=tuple(sample.match_id for sample in matches),
        references=tuple(np.asarray(sample.reference) for sample in matches),
        dead_x=tuple(dead_features(sample.observations)[0] for sample in matches),
        control_x=tuple(control_features_fn(sample.observations)[0] for sample in matches),
        swapped_x=tuple(control_features_fn(swap_teams(sample.observations))[0] for sample in matches)
        if augment else None,
        # Inputs without dead-ball supervision (e.g. SkillCorner 1ed50b7: possession only) never
        # enter the dead head's loss; their H/A frames still supervise the team head.
        masks_d=tuple((sample.observations.period_id > 0) & np.isin(sample.reference, ["H", "A", "D"])
                      & (sample.provenance.get("dead_supervision", True) is not False)
                      for sample in matches),
        masks_c=tuple((sample.observations.period_id > 0) & np.isin(sample.reference, ["H", "A"])
                      for sample in matches),
        weights=tuple(spell_weights(sample.reference, sample.observations.period_id, sample.observations.fps)
                      for sample in matches) if possession_weights else None)


def fit_predict_heads(inputs: HeadInputs, train: Sequence[int], predict: Sequence[int],
                      parameters: Mapping, *, recipe: str = "gru11",
                      periods: Sequence[np.ndarray] | None = None,
                      fps: Sequence[float] | float | None = None,
                      model_seed: int | None = None,
                      ensemble_seeds: Sequence[int] | None = None,
                      tta: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None
                      ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Heads fitted on ``train``: {index: (probability_H, probability_D)} for ``predict``.

    The selected GRU recipe uses the supplied features without recomputing them.
    Augmentation follows ``inputs.swapped_x``. GRU requires periods aligned to every
    input and one positive shared fps (a scalar or per-input sequence). Its scaler,
    loss and windows use training inputs only. All variants of a predicted match
    must be absent from training. Seeds follow ``run_lomo``; ensemble members are
    averaged in logit space. Models are discarded on return. ``inputs.weights`` weighs
    the loss rows. ``tta`` maps every predicted index to its (dead, control) features on
    exchanged teams (``swapped_features``); the ensemble's predictions on both are then
    combined by ``swap_average``.
    """
    train, predict = list(train), list(predict)
    if not train or not predict:
        raise ValueError("fitting heads requires training and predicted inputs")
    if {inputs.match_ids[i] for i in train} & {inputs.match_ids[i] for i in predict}:
        raise ValueError("a predicted match must never enter training")
    if tta is not None and set(tta) != set(predict):
        raise ValueError("test-time swap needs swapped features for exactly the predicted inputs")
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    learner = RECIPES[recipe][5]
    members = [dict(parameters) if seed is None else {**parameters, _seed_key(learner): seed}
               for seed in _seeds(model_seed, ensemble_seeds)]
    if periods is None or fps is None or len(periods) != len(inputs.match_ids):
        raise ValueError("GRU requires periods and fps aligned to every input")
    rates = np.asarray(fps)
    if rates.ndim == 0:
        rates = np.repeat(rates, len(inputs.match_ids))
    if (rates.shape != (len(inputs.match_ids),) or rates.dtype.kind not in "fiu"
            or not np.isfinite(rates).all() or (rates <= 0).any()
            or len(set(rates[train + predict])) != 1):
        raise ValueError("GRU inputs must share one positive finite frame rate")
    for i in train + predict:
        if np.asarray(periods[i]).shape != (len(inputs.dead_x[i]),):
            raise ValueError("GRU periods must align to input frames")
    from possession.learning.sequence import SequenceHead
    train_periods = [periods[i] for i in train]
    xs, ys = [inputs.control_x[i] for i in train], [inputs.references[i] == "H" for i in train]
    copies = 1
    if inputs.swapped_x is not None:
        xs += [inputs.swapped_x[i] for i in train]
        ys += [inputs.references[i] == "A" for i in train]
        copies = 2
    weights = None if inputs.weights is None else [inputs.weights[i] for i in train]
    outputs = {i: ([], [], [], []) for i in predict}
    with threadpool_limits(limits=RECIPES[recipe][4]):
        for member in members:
            dead_model = SequenceHead(member).fit(
                [inputs.dead_x[i] for i in train], [inputs.references[i] == "D" for i in train],
                [inputs.masks_d[i] for i in train], train_periods, float(rates[train[0]]),
                **({} if weights is None or not member.get("dead_loss_weights", True)
                   else {"weights": weights}))
            control_model = SequenceHead(member).fit(
                xs, ys, [inputs.masks_c[i] for i in train] * copies,
                train_periods * copies, float(rates[train[0]]),
                **({} if weights is None else {"weights": weights * copies}))
            for i in predict:
                views = [(inputs.dead_x[i], inputs.control_x[i])] + ([] if tta is None else [tta[i]])
                for offset, (dead_x, control_x) in enumerate(views):
                    ph = control_model.predict(control_x, periods[i])
                    pd = dead_model.predict(dead_x, periods[i])
                    outputs[i][2 * offset].append(ph)
                    outputs[i][2 * offset + 1].append(pd)
    results = {}
    for i, (ph, pd, swapped_h, swapped_d) in outputs.items():
        results[i] = (average_logits(ph), average_logits(pd))
        if tta is not None:
            results[i] = swap_average(*results[i], average_logits(swapped_h), average_logits(swapped_d))
    return results


def _seeds(model_seed: int | None, ensemble_seeds: Sequence[int] | None) -> list[int | None]:
    if ensemble_seeds is None:
        return [None if model_seed is None else int(model_seed)]
    seeds = list(ensemble_seeds)
    if model_seed is not None:
        raise ValueError("model_seed and ensemble_seeds are mutually exclusive")
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) for s in seeds)):
        raise ValueError("ensemble_seeds must be distinct integers")
    return [int(s) for s in seeds]


def fit_final(matches: Iterable[MatchInput], recipe: str = "gru11", model_seed: int | None = None,
              ensemble_seeds: Sequence[int] | None = None, parameter_overrides: Mapping | None = None,
              possession_weights: bool = False) -> dict:
    """Fit both heads once on every input (no match held out), one model per seed, for a final model.

    Same features, loss rows, augmentation, seeds and parameters as ``run_lomo``. Inputs
    are consumed one at a time and only their features, masks and periods are kept, so a
    generator lets each observation be freed. ``members`` keeps the seed order, which
    ``predict_members`` follows when averaging logits. ``possession_weights`` weighs the
    loss as in ``run_lomo``; test-time swap is a prediction option (``predict_members``).
    """
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    dead_features, control_features_fn, _, augment, threads, learner = RECIPES[recipe]
    parameters = recipe_parameters(recipe, parameter_overrides)
    members = [parameters if seed is None else {**parameters, _seed_key(learner): seed}
               for seed in _seeds(model_seed, ensemble_seeds)]
    parts, periods, fps, names = [], [], set(), None
    for sample in matches:
        validate_inputs([sample])
        if names is None:  # recorded with the model so prediction can refuse feature drift
            names = {"dead": list(dead_features(sample.observations)[1]),
                     "control": list(control_features_fn(sample.observations)[1])}
        parts.append(head_inputs([sample], recipe, possession_weights))
        periods.append(sample.observations.period_id)
        fps.add(float(sample.observations.fps))
    if not parts:
        raise ValueError("final training requires at least one input")
    if len(fps) != 1:
        raise ValueError("all training inputs must share one frame rate")
    rate = fps.pop()
    inputs = _concatenate(parts)
    del parts
    train = range(len(periods))
    yd = np.concatenate([(inputs.references[i][inputs.masks_d[i]] == "D").astype(np.int8) for i in train])
    yc = np.concatenate([(inputs.references[i][inputs.masks_c[i]] == side).astype(np.int8) for side in
                         (("H", "A") if augment else ("H",)) for i in train])
    fitted = []
    with threadpool_limits(limits=threads):
        from possession.learning.sequence import SequenceHead
        xs, ys = list(inputs.control_x), [inputs.references[i] == "H" for i in train]
        if augment:
            xs += list(inputs.swapped_x)
            ys += [inputs.references[i] == "A" for i in train]
        weights = {} if inputs.weights is None else {"weights": list(inputs.weights)}
        for member in members:
            dead_model = SequenceHead(member).fit(
                list(inputs.dead_x), [inputs.references[i] == "D" for i in train], list(inputs.masks_d),
                periods, rate, **(weights if member.get("dead_loss_weights", True) else {}))
            control_model = SequenceHead(member).fit(
                xs, ys, list(inputs.masks_c) * (2 if augment else 1), periods * (2 if augment else 1),
                rate, **{key: value * (2 if augment else 1) for key, value in weights.items()})
            fitted.append((member[_seed_key(learner)], dead_model, control_model))
    return {"recipe": recipe, "learner": learner, "parameters": parameters, "augment": augment,
            "possession_weights": bool(possession_weights),
            "threads": threads, "members": fitted, "inputs": len(periods), "feature_names": names,
            "training_rows": {"dead": int(yd.size), "control": int(yc.size)},
            "train_classes": {"dead": np.unique(yd).tolist(), "control": np.unique(yc).tolist()}}


def predict_members(members: Sequence[tuple], learner: str, dead_x: np.ndarray, control_x: np.ndarray,
                    period_id: np.ndarray, swapped: tuple[np.ndarray, np.ndarray] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(probability_H, probability_D): logits averaged over (dead, control) members in order.

    ``swapped`` (dead, control features of ``swapped_features``) adds the test-time team
    swap: the ensemble on both views is combined by ``swap_average``, as in ``run_lomo``.
    """
    if learner != "gru":
        raise ValueError("only GRU heads are supported")

    def ensemble(dead_x, control_x):
        dead, control = [], []
        for dead_model, control_model in members:
            dead.append(dead_model.predict(dead_x, period_id))
            control.append(control_model.predict(control_x, period_id))
        return average_logits(control), average_logits(dead)

    if swapped is None:
        return ensemble(dead_x, control_x)
    return swap_average(*ensemble(dead_x, control_x), *ensemble(*swapped))


def run_lomo(matches: Sequence[MatchInput], config: Mapping,
             scorer: Callable | None = None, recipe: str = "gru11",
             train_only: frozenset[str] = frozenset(), model_seed: int | None = None,
             ensemble_seeds: Sequence[int] | None = None,
             parameter_overrides: Mapping | None = None, tta: bool = False,
             possession_weights: bool = False, train_variants: Iterable[MatchInput] = ()) -> list[dict]:
    """Return full aligned OOF arrays and fold audit, with no artifact writes.

    Optional scorer(reference, prediction, period_id, fps) receives full arrays,
    including X and period zero. The common scorer owns interval/mask semantics.
    Caller owns any candidate selection; OOF output is development evidence.
    ``recipe`` picks features, model and augmentation from ``RECIPES``. With swap
    augmentation each training match also enters the H/A loss with teams exchanged
    and the pitch rotated; predictions always use the original observations.
    Matches in ``train_only`` enter every fold's training and are never held out.
    ``model_seed`` replaces the recipe's model seed (``random_state`` or ``seed``) to
    measure seed variance; by default the fixed seed of the recipe is used.
    ``ensemble_seeds`` trains one model per seed for each head (same features and
    folds) and averages their logits (``average_logits``); exclusive with ``model_seed``.
    ``parameter_overrides`` updates the recipe's model parameters (``recipe_parameters``).
    ``tta`` also predicts each held-out input with teams exchanged (``swapped_features``,
    computed for that input only) and combines both by ``swap_average``.
    ``possession_weights`` weighs every training frame by ``spell_weights``.
    ``train_variants`` are extra copies (e.g. other degradations) of loaded matches,
    consumed one at a time so only their features are kept: a copy trains every fold
    whose held-out match has a different ``match_id`` and is never predicted or scored.
    """
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    _, _, _, augment, threads, learner = RECIPES[recipe]
    parameters = recipe_parameters(recipe, parameter_overrides)
    members = [parameters if seed is None else {**parameters, _seed_key(learner): seed}
               for seed in _seeds(model_seed, ensemble_seeds)]
    samples = list(matches)
    for sample in samples:
        _validate(sample)
    groups = sorted({sample.match_id for sample in samples} - set(train_only))
    if len(groups) < 3:
        raise ValueError("LOMO requires at least three distinct match IDs")
    _costs(config)
    inputs = head_inputs(samples, recipe, possession_weights)
    periods = [sample.observations.period_id for sample in samples]
    fps = [sample.observations.fps for sample in samples]
    parts, loaded = [inputs], {sample.match_id for sample in samples}
    for variant in train_variants:  # one copy at a time: only features, masks and periods stay
        _validate(variant)
        if variant.match_id not in loaded:
            raise ValueError("a training variant must share the match_id of a loaded match")
        parts.append(head_inputs([variant], recipe, possession_weights))
        periods.append(variant.observations.period_id)
        fps.append(variant.observations.fps)
    variants = len(parts) - 1
    if variants:
        inputs = _concatenate(parts)
    del parts
    masks_d, masks_c, references = inputs.masks_d, inputs.masks_c, inputs.references
    folds = []
    # Bound native CPU threads for small repeatable development runs.
    with threadpool_limits(limits=threads):
        for held_out in groups:
            # Variants carry their source match_id, so the held-out match's copies stay out.
            train = [i for i, match_id in enumerate(inputs.match_ids) if match_id != held_out]
            yd = np.concatenate([(references[i][masks_d[i]] == "D").astype(np.int8) for i in train])
            yc = [(references[i][masks_c[i]] == "H").astype(np.int8) for i in train]
            if augment:  # exchanged teams: the same frames, labelled with the other team
                yc += [(references[i][masks_c[i]] == "A").astype(np.int8) for i in train]
            yc = np.concatenate(yc)
            # Only the evaluated copies (indices below len(samples)) are ever predicted.
            held = [i for i, sample in enumerate(samples) if sample.match_id == held_out]
            swapped = None if not tta else {
                i: swapped_features(samples[i].observations, recipe,
                                    None if inputs.swapped_x is None else inputs.swapped_x[i]) for i in held}
            outputs = fit_predict_heads(inputs, train, held, parameters, recipe=recipe, periods=periods,
                                        fps=fps, model_seed=model_seed, ensemble_seeds=ensemble_seeds,
                                        tta=swapped)
            del swapped
            fold = {"held_out_match_id": held_out,
                    "train_match_ids": sorted({inputs.match_ids[i] for i in train}),
                    "training_rows": {"dead": int(yd.size), "control": int(yc.size)},
                    "train_classes": {"dead": np.unique(yd).tolist(), "control": np.unique(yc).tolist()},
                    "model_seeds": [member[_seed_key(learner)] for member in members],
                    "predictions": []}
            if variants:
                fold["train_variant_inputs"] = sum(i >= len(samples) for i in train)
            for i in held:
                sample = samples[i]
                data = sample.observations
                ph, pd = outputs[i]
                labels = {
                    "learned_HA_D": decode_probabilities(ph, pd, data.period_id, config),
                }
                metrics = {} if scorer is None else {
                    name: scorer(sample.reference, prediction, data.period_id, data.fps)
                    for name, prediction in labels.items()
                }
                fold["predictions"].append({"input_index": i, "labels": labels,
                                            "probability_H": ph, "probability_D": pd,
                                            "metrics": metrics})
            folds.append(fold)
    return folds
