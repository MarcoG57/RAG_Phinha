import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.data.external import iter_matches
from possession.evaluation.external import degrade, score_predictions


def _long_observed(data, seconds_per_period=600):
    """Ten-minute halves with the ball always seen, built by tiling a synthetic fixture."""
    from dataclasses import replace
    n = 2 * seconds_per_period * data.fps + 50
    reps = -(-n // data.n_frames)
    tile = lambda values: np.concatenate([values] * reps)[:n].copy()
    period = np.zeros(n, dtype=np.int8)
    period[25:25 + seconds_per_period * data.fps] = 1
    period[n - 25 - seconds_per_period * data.fps:n - 25] = 2
    ball_xy = np.full((n, 2), .5, dtype=np.float32)
    return replace(data, frame=np.arange(n, dtype=np.int32), record_present=np.ones(n, dtype=bool),
                   period_id=period, ball_xy=ball_xy, ball_observed=np.ones(n, dtype=bool),
                   xy=tile(data.xy), speed=tile(data.speed), observed=tile(data.observed))


def _team_swap_fixture():
    """Every player observed and the ball next to one home player, so a cross-team swap is visible."""
    from dataclasses import replace
    data = _long_observed(next(iter_matches(ROOT / 'configs/external.synthetic.toml')).observations)
    xy = np.where(np.isnan(data.xy), data.xy[:1], data.xy)
    ball = np.tile(np.array([[.3, .32]], dtype=np.float32), (data.n_frames, 1))
    return replace(data, xy=xy, observed=np.ones_like(data.observed), speed=np.nan_to_num(data.speed, nan=1.),
                   ball_xy=ball)


class ExternalEvaluationTests(unittest.TestCase):
    def test_manifest_three_matches_and_separate_reference(self):
            matches = list(iter_matches(ROOT / 'configs/external.synthetic.toml'))
            self.assertEqual(len(matches), 3)
            self.assertEqual(len({match.match_id for match in matches}), 3)
            for match in matches:
                self.assertEqual(match.reference.shape, (match.observations.n_frames,))
                self.assertNotIn('reference', vars(match.observations))
                self.assertEqual(match.provenance['kind'], 'synthetic')

    def test_official_scorer_preserves_period_spells(self):
            reference = np.array(list('HHHH'))
            report = score_predictions(reference, reference, np.array([1, 1, 2, 2]), 25)
            self.assertEqual(report['n_gt_spells'], 2)
            self.assertEqual(report['spell_f1_mean'], 1)
            self.assertEqual(report['source_n_frames'], 4)
            self.assertEqual(report['period_boundary_sentinels'], 26)
            self.assertEqual(set(report['per_period']), {'1', '2'})
            self.assertNotIn('pairs', report['transitions'][25])
            self.assertNotIn('missed', report['per_period']['1']['transitions'][25])

    def test_unknown_reference_is_not_compacted_or_scored(self):
            reference = np.array(list('HHXAA'))
            report = score_predictions(reference, np.array(list('HHDAA')), np.ones(5, dtype=int), 25)
            self.assertEqual(report['n_graded'], 4)
            self.assertEqual(report['spell_f1_mean'], 1)
            self.assertEqual(report['source_n_frames'], 5)

    def test_non_native_fps_rejected(self):
            with self.assertRaisesRegex(ValueError, '25 fps'):
                score_predictions(['H'], ['H'], [1], 10)

    def test_malformed_manifest_and_noncanonical_reference_are_rejected(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest = folder / 'manifest.toml'
                manifest.write_text('[[matches]]\nid="one"\nformat="synthetic"\npath="fixture.json"\n')
                fixture = json.loads((ROOT / 'tests/fixtures/external/one.json').read_text())
                fixture['reference'][0] = 'Home'
                (folder / 'fixture.json').write_text(json.dumps(fixture))
                with self.assertRaisesRegex(ValueError, 'canonical'):
                    list(iter_matches(manifest))
                manifest.write_text('[[matches]]\nid="one"\nformat="synthetic"\npath="https://example.com/data"\n')
                with self.assertRaisesRegex(ValueError, 'local path'):
                    list(iter_matches(manifest))

    def test_dead_ball_missing_hides_most_stoppages_and_needs_the_reference(self):
            data = _long_observed(next(iter_matches(ROOT / 'configs/external.synthetic.toml')).observations)
            reference = np.where(data.period_id > 0, "H", "X")
            stops = [(s, s + 250) for s in range(100, data.n_frames - 400, 1000)]  # 10 s stoppages
            for start, end in stops:
                reference[start:end] = "D"
            with self.assertRaisesRegex(ValueError, "reference"):
                degrade(data, 'dead_ball_missing', 0)
            out = degrade(data, 'dead_ball_missing', 0, reference)
            hidden = [not out.ball_observed[s:e].any() for s, e in stops]
            self.assertGreater(np.mean(hidden), .6)  # 0.8 per stoppage
            self.assertLess(np.mean(hidden), 1.0)
            for (s, e), gone in zip(stops, hidden):
                if not gone:
                    continue
                np.testing.assert_array_equal(out.ball_observed[s:e], False)  # a whole stoppage, not a part
            live = (reference != "D") & (data.period_id > 0)
            self.assertAlmostEqual(float((~out.ball_observed[live]).mean()), .2, delta=.05)  # long gaps in play
            again = degrade(data, 'dead_ball_missing', 0, reference)
            np.testing.assert_array_equal(out.ball_observed, again.ball_observed)
            combined = degrade(data, 'combined_dead', 0, reference)
            self.assertTrue((~combined.observed).any())  # player dropout from combined is kept
            self.assertEqual(combined.provenance["degradation"]["scenario"], "combined_dead")

    def test_official_false_abstention_is_still_a_miss(self):
            report = score_predictions(np.array(list('HHAA')), np.array(list('XXXX')), np.ones(4, dtype=int), 25)
            self.assertEqual(report['n_abstain_on_graded'], 4)
            self.assertEqual(report['accuracy'], 0)

    def test_transition_matching_cannot_pair_different_periods(self):
            report = score_predictions(np.array(list('HHAHHH')), np.array(list('HHHHHA')),
                                       np.array([1, 1, 1, 2, 2, 2]), 25)
            self.assertEqual(report['transitions'][25]['n_matched'], 0)
