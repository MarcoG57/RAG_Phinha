"""Only invented players/coordinates; no external data is redistributed."""
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.data.sportec import load_sportec
from possession.data.atd import AtdMeta
from possession.data.external import empty_observations

META = '''<PutDataRequest><MatchInformation>
<General MatchId="toy"/><Environment PitchX="100" PitchY="60"/>
<Teams><Team TeamId="away" TeamName="Visitors" Role="guest"><Players>
<Player PersonId="a"/></Players></Team>
<Team TeamId="home" TeamName="Hosts" Role="home"><Players>
<Player PersonId="h"/><Player PersonId="sub"/></Players></Team></Teams>
</MatchInformation></PutDataRequest>'''


def positions():
    rows = ['<Positions><MetaData MatchId="toy" Type="pitch-size"><PitchSize X="100" Y="60"/></MetaData>']
    for section, ids in [('firstHalf', [10000, 10002]), ('secondHalf', [10005, 10006])]:
        for team, person, x in [('home', 'h', '-49'), ('away', 'a', '40'), ('BALL', 'ball', '-50')]:
            rows.append(f'<FrameSet MatchId="toy" GameSection="{section}" TeamId="{team}" PersonId="{person}">')
            for i in ids:
                gt = 'BallStatus="1" BallPossession="1"' if i != 10002 else 'BallStatus="1" BallPossession="9"'
                rows.append(f'<Frame N="{i}" X="{x}" Y="30" S="36" {gt if team == "BALL" else ""}/>')
            rows.append('</FrameSet>')
    return ''.join(rows) + '</Positions>'


class SportecTests(unittest.TestCase):
    def test_dense_budget_rejects_before_array_allocation(self):
        for frames, slots in ((2_000_000, 10_000), (2_000_000, 40), (10, 65)):
            meta = AtdMeta('H', 'A', np.array(['H'] * slots))
            with self.subTest(frames=frames, slots=slots), \
                    patch('possession.data.external.np.arange') as allocate, \
                    patch('possession.data.external.np.full') as full, \
                    patch('possession.data.external.np.zeros') as zeros:
                with self.assertRaisesRegex(ValueError, 'slots|budget'):
                    empty_observations('toy', 'sportec', 25, (105, 68), meta, frames)
                allocate.assert_not_called()
                full.assert_not_called()
                zeros.assert_not_called()

    def load(self, raw=None, meta=META):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'meta.xml').write_text(meta, encoding='utf-8')
            (p / 'positions.xml').write_text(raw or positions(), encoding='utf-8')
            return load_sportec(p / 'meta.xml', p / 'positions.xml', match_id='toy',
                                fps=25, coordinate_origin='center', coordinate_units='m', speed_units='km/h')

    def test_units_teams_unknowns_and_noncompacted_gaps(self):
        match = self.load()
        data = match.observations
        np.testing.assert_array_equal(data.frame, np.arange(7))
        np.testing.assert_array_equal(data.period_id, [1, 1, 1, 0, 0, 2, 2])
        np.testing.assert_array_equal(match.reference, list('HXXXXHH'))
        self.assertEqual(data.meta.slot_team.tolist(), ['H', 'H', 'A'])
        np.testing.assert_allclose(data.ball_xy[0], [0, 1])
        self.assertAlmostEqual(float(data.xy[0, 0, 0]), .01)
        self.assertEqual(float(data.speed[0, 0]), 10)
        self.assertTrue(np.isnan(data.xy[0, 1]).all())
        self.assertFalse(data.record_present[1])

    def test_ground_truth_does_not_change_observations(self):
        first = self.load()
        second = self.load(positions().replace('BallPossession="1"', 'BallPossession="2"').replace('BallStatus="1"', 'BallStatus="0"'))
        self.assertFalse(np.array_equal(first.reference, second.reference))
        for name in ('xy', 'speed', 'ball_xy', 'period_id', 'observed'):
            np.testing.assert_equal(getattr(first.observations, name), getattr(second.observations, name))
        self.assertEqual(first.observations.provenance, second.observations.provenance)

    def test_duplicate_key_is_rejected(self):
        raw = positions().replace('<Frame N="10000"', '<Frame N="10002"', 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load(raw)

    def test_unknown_team_or_period_is_rejected(self):
        for old, new in [('TeamId="home"', 'TeamId="mystery"'), ('firstHalf', 'mystery')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.load(positions().replace(old, new))

    def test_pitch_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pitch'):
            self.load(positions().replace('X="100" Y="60"', 'X="105" Y="68"'))

    def test_missing_or_unrecognized_status_is_unknown_and_dead_is_explicit(self):
        raw = positions().replace('BallStatus="1"', 'BallStatus="9"')
        self.assertEqual(set(self.load(raw).reference), {'X'})
        raw = positions().replace('BallStatus="1"', 'BallStatus="0"')
        self.assertEqual(self.load(raw).reference.tolist(), list('DXDXXDD'))

    def test_missing_coordinate_is_nan_and_not_zero(self):
        match = self.load(positions().replace('X="-49"', 'X=""', 1))
        self.assertFalse(match.observations.observed[0, 0])
        self.assertTrue(np.isnan(match.observations.xy[0, 0]).all())

    def test_conflicting_period_ranges_require_alignment(self):
        with self.assertRaisesRegex(ValueError, 'period frame counters'):
            self.load(positions().replace('N="10005"', 'N="10001"'))

    def test_malformed_xml_returns_clear_input_error(self):
        with self.assertRaisesRegex(ValueError, 'malformed Sportec XML'):
            self.load('<Positions>')


if __name__ == '__main__':
    unittest.main()
