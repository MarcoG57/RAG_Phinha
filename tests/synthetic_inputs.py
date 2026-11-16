import numpy as np
from possession.data.atd import AtdData, AtdMeta

def synthetic_atd():
    """Small, deterministic input fixture with no reference labels."""
    n = 60
    frame = np.arange(n, dtype=np.int32)
    period = np.array([0] * 2 + [1] * 20 + [0] * 2 + [2] * 36, dtype=np.int8)
    ball = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (n, 1))
    xy = np.tile(np.array([[[0.5, 0.5], [0.8, 0.8]]], dtype=np.float32), (n, 1, 1))
    observed = np.ones((n, 2), dtype=bool)
    return AtdData(
        dataset_id="SYNTHETIC",
        source="tactical",
        fps=25,
        pitch=(100.0, 50.0),
        meta=AtdMeta("Home FC", "Away FC", np.array(["H", "A"])),
        frame=frame,
        record_present=np.ones(n, dtype=bool),
        period_id=period,
        ball_xy=ball,
        ball_observed=np.ones(n, dtype=bool),
        xy=xy,
        speed=np.zeros((n, 2), dtype=np.float32),
        observed=observed,
        periods_known=True,
        provenance={
            "meta_file": "synthetic_meta.xml",
            "raw_file": "synthetic_raw.txt",
            "n_records": n,
            "frame_id_min": 1,
            "frame_id_max": n,
            "frame_ids_contiguous": True,
            "frame_ids_sorted": True,
        },
    )


def make_data(ball, players, observed, slot_team="HHA", pitch=(100.0, 50.0)):
    """AtdData mínimo: ``ball`` (N,2) y ``players`` (N,S,2) normalizados."""
    ball = np.asarray(ball, dtype=np.float32)
    xy = np.asarray(players, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    xy = np.where(observed[..., None], xy, np.nan).astype(np.float32)
    n = ball.shape[0]
    return AtdData(
        dataset_id="T", source="tactical", fps=25, pitch=pitch,
        meta=AtdMeta("H FC", "A FC", np.array(list(slot_team))),
        frame=np.arange(n, dtype=np.int32), record_present=np.ones(n, bool),
        period_id=np.ones(n, dtype=np.int8), ball_xy=ball,
        ball_observed=np.isfinite(ball).all(axis=1), xy=xy,
        speed=np.zeros(observed.shape, np.float32), observed=observed, periods_known=True,
    )
