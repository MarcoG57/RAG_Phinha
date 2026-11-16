"""The selected mixdead decoder: learned evidence plus fixed dead-ball evidence."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from possession.data.atd import AtdData
from possession.features.play_state import PlayStateConfig, dead_evidence
from possession.learning.workflow import _evidence, decode_evidence

VARIANTS = ("mixdead",)
COMPONENTS = ("mixdead",)



def variant_names(components: Iterable[str] | None) -> list[str]:
    """Select the retained mixdead variant, rejecting removed experimental variants."""
    chosen = list(components or [])
    if len(set(chosen)) != len(chosen) or not set(chosen) <= set(COMPONENTS):
        raise ValueError(f"decode variants must be distinct values of {COMPONENTS}")
    return [name for name in VARIANTS if name in chosen]


def needs_hand_evidence(variant: str) -> bool:
    """The selected mixdead variant requires the card's fixed hand-evidence config."""
    return variant == "mixdead"


def variant_evidence(probability_h: np.ndarray, probability_d: np.ndarray, data: AtdData,
                     config: Mapping, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Learned team evidence and the sum of learned and hand dead evidence."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown decode variant: {variant}")
    control_e, dead_e = _evidence(probability_h), _evidence(probability_d)
    if control_e.shape != (data.n_frames,) or dead_e.shape != (data.n_frames,):
        raise ValueError("probabilities must align one value per frame")
    if "play_state" not in config:
        raise ValueError(f"{variant} decoding requires the play_state config")
    hand_dead = dead_evidence(data, PlayStateConfig(**config["play_state"]))
    dead_e = dead_e + hand_dead
    return control_e, dead_e


def decode_variant(probability_h: np.ndarray, probability_d: np.ndarray, data: AtdData, config: Mapping,
                   variant: str, known_dead: np.ndarray | None = None) -> np.ndarray:
    """Decode mixdead evidence using the selected joint H/A/D Viterbi."""
    return decode_evidence(*variant_evidence(probability_h, probability_d, data, config, variant),
                           data.period_id, config, known_dead)
