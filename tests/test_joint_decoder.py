"""Joint H/A/D decoder: limit cases equal the two-stage decoder; explicit joint structure."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from possession.inference.play_state_viterbi import decode_joint  # noqa: E402
from possession.learning.workflow import decode_evidence  # noqa: E402


def signals(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    period = np.r_[np.zeros(10, int), np.ones(n // 2, int), np.full(n - n // 2 - 10, 2)]
    control = np.repeat(rng.normal(0, 2, n // 50 + 1), 50)[:n] + rng.normal(0, 1, n)
    dead = np.repeat(rng.normal(-1, 3, n // 100 + 1), 100)[:n] + rng.normal(0, 1, n)
    return control, dead, period


class JointDecoderTests(unittest.TestCase):
    def test_unambiguous_constant_evidence_and_outside_periods(self):
        period = np.array([0, 1, 1, 1, 2, 2, 0])
        for control, dead, expected in ((20., -20., "XHHHHHX"),
                                        (-20., -20., "XAAAAAX"),
                                        (0., 20., "XDDDDDX")):
            with self.subTest(expected=expected):
                labels = decode_joint(np.full(7, control), np.full(7, dead), period, 0., 0.)
                self.assertEqual("".join(labels), expected)

    def test_known_dead_frames_are_forced_to_d(self):
        control, dead, period = signals(seed=2)
        known = np.zeros(control.size, bool)
        known[100:400] = True
        labels = decode_joint(np.abs(control) + 5, np.full(control.size, -20.0), period, 50.0, 50.0, known)
        self.assertTrue((labels[known & (period > 0)] == "D").all())

    def test_mixdead_adds_hand_dead_evidence_to_learned_log_odds(self):
        from synthetic_inputs import synthetic_atd
        from possession.features.play_state import PlayStateConfig, dead_evidence
        from possession.learning.decoding import variant_evidence
        data = synthetic_atd()
        data.ball_observed[20:40] = False
        home = np.full(data.n_frames, .8)
        dead = np.full(data.n_frames, .3)
        control_e, dead_e = variant_evidence(home, dead, data, {"play_state": {}}, "mixdead")
        np.testing.assert_allclose(control_e, 0.5 * np.log(.8 / .2))
        np.testing.assert_allclose(dead_e, 0.5 * np.log(.3 / .7) + dead_evidence(data, PlayStateConfig()))
        with self.assertRaisesRegex(ValueError, "play_state"):
            variant_evidence(home, dead, data, {}, "mixdead")

    def test_unknown_structure_is_rejected(self):
        control, dead, period = signals(seed=5)
        config = {"decoder": {"switch_cost": 50.0, "structure": "triple"}, "dead_decoder": {"switch_cost": 50.0}}
        with self.assertRaisesRegex(ValueError, "decoder structure"):
            decode_evidence(control, dead, period, config)

    def test_predict_final_passes_the_card_structure(self):
        from predict_final import decoder_config
        card = {"decoder": {"control_switch_cost": 50.0, "dead_switch_cost": 50.0,
                            "structure": "joint", "variant": "mixdead",
                            "hand_evidence": {"control": {}, "play_state": {}}}}
        self.assertEqual(decoder_config(card)["decoder"]["structure"], "joint")


if __name__ == "__main__":
    unittest.main()
