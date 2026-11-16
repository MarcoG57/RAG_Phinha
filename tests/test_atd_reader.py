"""Lector ATD y QA con un ATD sintético generado en el test; no lee material del partido."""
import importlib.util
import os
import sys
import tempfile
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from unittest import mock

if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("pandas") is None:
    raise unittest.SkipTest("numpy/pandas ausentes; instalar requirements.lock.txt (README.md)")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from possession.data import atd  # noqa: E402
from possession.data.atd import load_atd, read_atd, read_meta  # noqa: E402
from inference_support import validate_atd_alignment  # noqa: E402

PITCH = (100.0, 50.0)
META = """<?xml version="1.0"?>
<Root><Teams>
  <Team id="T-home"><Name>Local FC</Name></Team>
  <Team id="T-away"><Name>Visitante CF</Name></Team>
</Teams><Players>
  <Player id="P1" teamId="T-home"/>
  <Player id="P2" teamId="T-away"/>
  <Player id="P3" teamId="T-home"/>
</Players></Root>
"""
NAN3 = "NaN,NaN,NaN"
# frame_id 1..6; el balón falta en 3 y 4; el slot 2 falta en 2; el slot 3 sólo aparece en 5.
RAW = "\n".join([
    f"1:0.1,0.2,1.0;0.5,0.5,2.0;{NAN3}:0.1,0.2,0.0",
    f"2:0.2,0.2,1.5;{NAN3};{NAN3}:0.2,0.2,0.0",
    f"3:0.3,0.2,1.5;0.5,0.6,2.0;{NAN3}:NaN,NaN,NaN",
    f"4:0.4,0.2,1.5;0.5,0.7,2.0;{NAN3}:NaN,NaN,NaN",
    "5:0.5,0.2,1.5;0.5,0.8,2.0;0.9,0.9,3.0:0.5,0.25,0.0",
    f"6:0.6,0.2,1.5;0.5,0.9,2.0;{NAN3}:0.6,0.2,0.0",
]) + "\n"


class AtdFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.meta = self.write("meta.xml", META)
        self.raw = self.write("raw.txt", RAW)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def read(self, raw=None, **kwargs):
        options = {"source": "tactical", "pitch": PITCH, "fps": 25}
        options.update(kwargs)
        return read_atd(self.meta, raw or self.raw, **options)


class MetaTests(AtdFixture):
    def test_first_team_is_home_and_slots_follow_player_order(self):
        meta = read_meta(self.meta)
        self.assertEqual((meta.home_name, meta.away_name), ("Local FC", "Visitante CF"))
        self.assertEqual(meta.slot_team.tolist(), ["H", "A", "H"])
        self.assertEqual(meta.n_slots, 3)

    def test_unknown_team_and_wrong_team_count_are_rejected(self):
        bad_team = self.write("bad.xml", META.replace('teamId="T-away"', 'teamId="T-x"'))
        with self.assertRaisesRegex(ValueError, "teamId"):
            read_meta(bad_team)
        one_team = self.write("one.xml", META.replace('<Team id="T-away"><Name>Visitante CF</Name></Team>', ""))
        with self.assertRaisesRegex(ValueError, "2 equipos"):
            read_meta(one_team)
        no_players = self.write("none.xml", '<Root><Team id="a"/><Team id="b"/></Root>')
        with self.assertRaisesRegex(ValueError, "Player"):
            read_meta(no_players)


class AlignmentTests(AtdFixture):
    def test_frame_is_frame_id_minus_one(self):
        data = self.read()
        self.assertEqual(data.n_frames, 6)
        self.assertEqual(data.frame.tolist(), list(range(6)))
        self.assertAlmostEqual(float(data.xy[0, 0, 0]), 0.1, places=6)  # frame_id 1 -> frame 0
        self.assertAlmostEqual(float(data.xy[5, 0, 0]), 0.6, places=6)
        self.assertTrue(data.provenance["frame_ids_contiguous"])
        np.testing.assert_allclose(data.time_s, np.arange(6) / 25)

    def test_gap_in_frame_ids_is_flagged_not_shifted(self):
        lines = RAW.splitlines()
        raw = self.write("gap.txt", "\n".join(lines[:2] + lines[3:]) + "\n")
        data = self.read(raw, n_frames=6)
        self.assertFalse(data.provenance["frame_ids_contiguous"])
        self.assertEqual(data.record_present.tolist(), [True, True, False, True, True, True])
        self.assertFalse(data.observed[2].any())
        self.assertAlmostEqual(float(data.xy[3, 0, 0]), 0.4, places=6)  # no se desplaza
        with self.assertRaises(ValueError):
            validate_atd_alignment(data)

    def test_unsorted_records_land_on_their_own_frame(self):
        lines = RAW.splitlines()
        raw = self.write("shuffled.txt", "\n".join(reversed(lines)) + "\n")
        data = self.read(raw)
        self.assertFalse(data.provenance["frame_ids_sorted"])
        np.testing.assert_array_equal(data.xy, self.read().xy)

    def test_duplicate_and_out_of_range_ids_are_rejected(self):
        dup = self.write("dup.txt", RAW + RAW.splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicados"):
            self.read(dup)
        with self.assertRaisesRegex(ValueError, "fuera de n_frames"):
            self.read(n_frames=4)
        zero = self.write("zero.txt", RAW.replace("1:0.1", "0:0.1", 1))
        with self.assertRaisesRegex(ValueError, "frame_id < 1"):
            self.read(zero)

    def test_periods_cover_inclusive_bounds_and_outside_is_zero(self):
        data = self.read(periods=[(1, 1, 2), (2, 4, 5)])
        self.assertEqual(data.period_id.tolist(), [0, 1, 1, 0, 2, 2])
        self.assertTrue(data.periods_known)
        self.assertFalse(self.read().periods_known)
        self.assertEqual(self.read().period_id.tolist(), [0] * 6)


class MissingValueTests(AtdFixture):
    def test_absence_is_nan_with_false_mask_never_zero(self):
        data = self.read()
        self.assertEqual(data.ball_observed.tolist(), [True, True, False, False, True, True])
        self.assertTrue(np.isnan(data.ball_xy[2:4]).all())
        self.assertFalse(data.observed[1, 1])
        self.assertTrue(np.isnan(data.xy[1, 1]).all())
        self.assertTrue(np.isnan(data.speed[1, 1]))
        self.assertEqual(data.observed[:, 2].tolist(), [False, False, False, False, True, False])

    def test_partial_coordinate_counts_as_missing(self):
        raw = self.write("partial.txt", RAW.replace("0.5,0.5,2.0", "0.5,NaN,2.0", 1))
        data = self.read(raw)
        self.assertFalse(data.observed[0, 1])
        self.assertTrue(np.isnan(data.xy[0, 1]).all())


class MalformedTests(AtdFixture):
    def test_malformed_lines_raise_with_line_number(self):
        cases = {
            "campos": "1:0.1,0.2,1.0;0.5,0.5,2.0;NaN,NaN,NaN\n",
            "grupos": "1:0.1,0.2,1.0;0.5,0.5,2.0:0.1,0.2,0\n",
            "no entero": "x:0.1,0.2,1.0;0.5,0.5,2.0;NaN,NaN,NaN:0.1,0.2,0\n",
            "no numérico": "1:0.1,abc,1.0;0.5,0.5,2.0;NaN,NaN,NaN:0.1,0.2,0\n",
            "3 valores": "1:0.1,0.2;0.5,0.5,2.0;NaN,NaN,NaN:0.1,0.2,0\n",
            "balón": "1:0.1,0.2,1.0;0.5,0.5,2.0;NaN,NaN,NaN:0.1\n",
        }
        for message, text in cases.items():
            with self.subTest(message=message):
                raw = self.write("bad.txt", text)
                with self.assertRaisesRegex(ValueError, message):
                    self.read(raw)

    def test_misallocated_groups_with_valid_total_are_rejected(self):
        # Revisión PR #17: grupos de 2/4/3 valores suman 9 y antes pasaban mezclando slots.
        raw = self.write("shifted.txt", "1:0.1,0.2;0.3,0.4,0.5,0.6;0.7,0.8,0.9:0.1,0.2,0\n")
        with self.assertRaisesRegex(ValueError, "slot 1 debe tener 3 valores"):
            self.read(raw)
        raw = self.write("shifted2.txt", "1:0.1,0.2,0.3;0.4,0.5,0.6,0.7;0.8,0.9:0.1,0.2,0\n")
        with self.assertRaisesRegex(ValueError, "slot 2 debe tener 3 valores"):
            self.read(raw)

    def test_empty_file_is_rejected_and_blank_lines_skipped(self):
        with self.assertRaisesRegex(ValueError, "sin registros"):
            self.read(self.write("empty.txt", "\n\n"))
        self.assertEqual(self.read(self.write("blank.txt", "\n" + RAW + "\n")).n_frames, 6)


class TableTests(AtdFixture):
    def test_frames_table_scales_each_axis_separately(self):
        frames = self.read().to_frames_df()
        self.assertEqual(list(frames.columns), [
            "dataset_id", "source", "frame", "time_s", "period_id",
            "ball_x_m", "ball_y_m", "ball_observed"])
        row = frames.iloc[4]
        self.assertAlmostEqual(row.ball_x_m, 0.5 * PITCH[0], places=4)
        self.assertAlmostEqual(row.ball_y_m, 0.25 * PITCH[1], places=4)
        self.assertTrue(np.isnan(frames.iloc[2].ball_x_m))

    def test_tracks_table_keeps_team_and_slot_without_inventing_rows(self):
        data = self.read()
        tracks = data.to_tracks_df()
        self.assertEqual(len(tracks), int(data.observed.sum()))
        self.assertTrue(tracks.observed.all())
        slot3 = tracks[tracks.slot_id == 3]
        self.assertEqual(slot3.frame.tolist(), [4])
        self.assertEqual(slot3.team.tolist(), ["H"])
        self.assertAlmostEqual(slot3.x_m.iloc[0], 0.9 * PITCH[0], places=4)
        self.assertAlmostEqual(slot3.y_m.iloc[0], 0.9 * PITCH[1], places=4)
        full = data.to_tracks_df(observed_only=False)
        self.assertEqual(len(full), 6 * 3)
        self.assertEqual(int(full.observed.sum()), len(tracks))


class LoadAtdTests(AtdFixture):
    """load_atd con un games.py simulado: rutas, partes y caché sin el paquete real."""

    def fake_games(self):
        games = mock.Mock()
        games.GAME = "TEST"
        games.DEFAULT_SOURCE = "tactical"
        games.GAMES = {"TEST": {"pitch": PITCH, "fps": 25}}
        games.require.return_value = [str(self.meta), str(self.raw)]
        games.half_video_blocks.return_value = [(1, 0, 2), (2, 3, 5)]
        games.source_frames.return_value = 6
        return games

    def test_concurrent_cache_writers_publish_complete_archives(self):
        data = self.read()
        destination = self.dir / "cache.npz"
        saved = Barrier(2)
        writer = Lock()
        real_savez = np.savez
        real_replace = os.replace

        def overlapping_save(*args, **kwargs):
            # Both writes finish before either publishes. The filesystem and
            # archive contents are real; only the interleaving is controlled.
            with writer:
                real_savez(*args, **kwargs)
            saved.wait(timeout=10)

        def serialized_publish(*args, **kwargs):
            with writer:
                return real_replace(*args, **kwargs)

        with mock.patch.object(atd.np, "savez", side_effect=overlapping_save), \
                mock.patch.object(atd.os, "replace", side_effect=serialized_publish):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [pool.submit(atd._save_cache, destination, data, key)
                           for key in ("first", "second")]
                for result in results:
                    result.result(timeout=15)

        with np.load(destination, allow_pickle=False) as archive:
            self.assertIn(str(archive["key"]), ("first", "second"))
            np.testing.assert_array_equal(archive["xy"], data.xy)
        self.assertEqual(list(self.dir.glob("*.npz")), [destination])

    def test_corrupt_cache_is_rebuilt_from_validated_inputs(self):
        games = self.fake_games()
        cache_dir = self.dir / "cache"
        with mock.patch.object(atd, "official_games", return_value=games), \
                mock.patch.object(atd, "CACHE_DIR", cache_dir):
            expected = load_atd("tactical")
            cache_path = cache_dir / "atd_TEST_tactical.npz"
            valid_bytes = cache_path.read_bytes()
            for payload in (b"not an archive", valid_bytes[:len(valid_bytes) // 2]):
                with self.subTest(payload_size=len(payload)):
                    cache_path.write_bytes(payload)
                    actual = load_atd("tactical")
                    np.testing.assert_array_equal(actual.xy, expected.xy)
                    np.testing.assert_array_equal(actual.period_id, expected.period_id)
            # A bad cache must not bypass validation of the authoritative input.
            cache_path.write_bytes(b"not an archive")
            self.raw.write_text("invalid tracking\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_atd("tactical")

    def test_open_cache_reader_does_not_prevent_loading_changed_inputs(self):
        games = self.fake_games()
        cache_dir = self.dir / "cache"
        with mock.patch.object(atd, "official_games", return_value=games), \
                mock.patch.object(atd, "CACHE_DIR", cache_dir):
            load_atd("tactical")
            cache_path = cache_dir / "atd_TEST_tactical.npz"
            self.raw.write_text(RAW.replace("0.6,0.2,1.5", "0.75,0.2,1.5"), encoding="utf-8")
            with cache_path.open("rb"), warnings.catch_warnings(record=True) as notices:
                actual = load_atd("tactical")
            self.assertAlmostEqual(float(actual.xy[5, 0, 0]), 0.75, places=6)
            if os.name == "nt":
                self.assertTrue(any("without caching" in str(n.message) for n in notices))
            self.assertEqual(list(cache_dir.glob("*.npz")), [cache_path])

    def test_cache_round_trip_and_invalidation(self):
        games = self.fake_games()
        with mock.patch.object(atd, "official_games", return_value=games), \
                mock.patch.object(atd, "CACHE_DIR", self.dir / "cache"):
            first = load_atd("tactical")
            self.assertTrue((self.dir / "cache" / "atd_TEST_tactical.npz").is_file())
            with mock.patch.object(atd, "read_atd", side_effect=AssertionError("no debe releer")):
                cached = load_atd("tactical")
            for name in ("xy", "speed", "ball_xy", "observed", "ball_observed", "period_id", "record_present"):
                np.testing.assert_array_equal(getattr(first, name), getattr(cached, name))
            self.assertEqual(cached.provenance, first.provenance)
            self.assertEqual(cached.meta.slot_team.tolist(), ["H", "A", "H"])
            self.assertEqual(cached.period_id.tolist(), [1, 1, 1, 2, 2, 2])

            self.raw.write_text(RAW.replace("0.6,0.2,1.5", "0.7,0.2,1.5"), encoding="utf-8")
            reread = load_atd("tactical")
            self.assertAlmostEqual(float(reread.xy[5, 0, 0]), 0.7, places=6)

    def test_legacy_cache_does_not_bypass_float32_validation(self):
        games = self.fake_games()
        cache_dir = self.dir / "cache"
        config = {
            "dataset_id": "TEST",
            "source": "tactical",
            "pitch": [100.0, 50.0],
            "fps": 25,
            "n_frames": 6,
            "periods": [(1, 0, 2), (2, 3, 5)],
        }
        cache_path = cache_dir / "atd_TEST_tactical.npz"

        with mock.patch.object(atd, "official_games", return_value=games), \
                mock.patch.object(atd, "CACHE_DIR", cache_dir):
            # Simula la caché v2 ya creada antes del validador float32.
            with mock.patch.object(atd, "CACHE_VERSION", 2):
                load_atd("tactical")
            with np.load(cache_path, allow_pickle=False) as cached_archive:
                legacy_key = str(cached_archive["key"])

            original_stat = self.raw.stat()
            invalid_raw = RAW.replace("0.5,0.25,0.0", "0.5,1e40,0.0", 1)
            self.raw.write_text(invalid_raw, encoding="utf-8")
            self.assertEqual(self.raw.stat().st_size, original_stat.st_size)
            os.utime(self.raw, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

            # El contenido cambió, pero el formato de clave v2 sigue identificando
            # la misma ruta, tamaño y mtime; esto acredita que la caché sí coincide.
            with mock.patch.object(atd, "CACHE_VERSION", 2):
                self.assertEqual(
                    atd._cache_key([str(self.meta), str(self.raw)], config),
                    legacy_key,
                )

            with self.assertRaisesRegex(ValueError, "rango float32"):
                load_atd("tactical")

    def test_cache_is_not_shared_between_data_dirs_with_identical_stat(self):
        # Revisión PR #17: mismo nombre, tamaño y mtime en dos carpetas devolvía la primera.
        other = self.dir / "other"
        other.mkdir()
        meta2 = other / "meta.xml"
        raw2 = other / "raw.txt"
        meta2.write_text(META, encoding="utf-8")
        raw2.write_text(RAW.replace("1:0.1,0.2,1.0", "1:0.9,0.2,1.0"), encoding="utf-8")
        for src, dst in ((self.meta, meta2), (self.raw, raw2)):
            stat = src.stat()
            os.utime(dst, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(self.raw.stat().st_size, raw2.stat().st_size)

        games = self.fake_games()
        with mock.patch.object(atd, "official_games", return_value=games), \
                mock.patch.object(atd, "CACHE_DIR", self.dir / "cache"):
            first = load_atd("tactical")
            games.require.return_value = [str(meta2), str(raw2)]
            second = load_atd("tactical")
        self.assertAlmostEqual(float(first.xy[0, 0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(second.xy[0, 0, 0]), 0.9, places=6)

    def test_cache_is_invalidated_by_effective_config(self):
        games = self.fake_games()
        with mock.patch.object(atd, "official_games", return_value=games), \
                mock.patch.object(atd, "CACHE_DIR", self.dir / "cache"):
            load_atd("tactical")
            games.GAMES = {"TEST": {"pitch": (120.0, 80.0), "fps": 50}}
            games.half_video_blocks.return_value = [(1, 0, 1), (2, 4, 5)]
            changed = load_atd("tactical")
            self.assertEqual(changed.pitch, (120.0, 80.0))
            self.assertEqual(changed.fps, 50)
            self.assertEqual(changed.period_id.tolist(), [1, 1, 0, 0, 2, 2])
            games.source_frames.return_value = 8
            self.assertEqual(load_atd("tactical").n_frames, 8)

    def test_missing_official_checkout_is_explicit(self):
        with mock.patch.object(atd, "OFFICIAL_DIR", self.dir / "no-official"):
            with self.assertRaisesRegex(FileNotFoundError, "checkout oficial"):
                atd.official_games()

    @unittest.skipUnless((atd.OFFICIAL_DIR / "games.py").is_file(), "falta .official")
    def test_official_games_is_importable(self):
        games = atd.official_games()
        self.assertEqual(games.source_frames("tactical"), 199319)
        self.assertEqual(games.GAMES[games.GAME]["fps"], 25)

    def test_broadcast_has_no_known_periods(self):
        games = self.fake_games()
        with mock.patch.object(atd, "official_games", return_value=games):
            data = load_atd("broadcast", use_cache=False)
        self.assertFalse(data.periods_known)
        games.half_video_blocks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
