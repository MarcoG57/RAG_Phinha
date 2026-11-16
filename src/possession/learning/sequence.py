"""Bidirectional GRU heads over per-frame features (PyTorch, optional dependency).

Same contract as the gradient-boosting heads: one binary head per question (dead vs
alive, home vs away), trained on known frames of the training matches only, returning a
probability per frame of the held-out match. The network sees the match at 5 Hz
(every ``STRIDE``-th frame) inside one period at a time, in windows of
``window_s`` seconds; probabilities are averaged over overlapping windows and
interpolated back to every frame. Hyperparameters are fixed before any score is seen.

NaN features become 0 plus a missing-value indicator; standardisation uses training
frames only. Training may run on a GPU; ``device="cpu"`` with one thread is the
reproducible path for a submission.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
import platform

import hashlib
import io
from pathlib import Path

import numpy as np

STRIDE = 5
SEQUENCE_PARAMETERS = {"hidden": 64, "layers": 2, "dropout": 0.2, "window_s": 60.0, "batch": 64,
                       "steps": 4000, "learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 20261029}
# gru11b (8 Nov audit): same network and training; the dead head ignores possession-duration
# weights and prediction averages 75 %-overlapping windows with triangular logit weights.
SEQUENCE_PARAMETERS_AUDIT = {**SEQUENCE_PARAMETERS, "window_overlap": 4, "triangular_average": True,
                             "dead_loss_weights": False}


def _torch():
    import torch
    return torch


def _cpu_reproducible(method):
    """Use the reproducible CPU path locally, then restore the caller's settings.

    This fixes threads and deterministic kernels within one environment; it does
    not promise identical floats across PyTorch releases or different hardware.
    """
    @wraps(method)
    def run(self, *args, **kwargs):
        torch = _torch()
        if str(self.device).split(":")[0] != "cpu":
            return method(self, *args, **kwargs)
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            return method(self, *args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.set_num_threads(threads)
    return run


def _encode(x: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(x)
    z = np.where(missing, 0.0, (np.nan_to_num(x) - centre) / scale)
    return np.concatenate((z, missing), axis=1).astype(np.float32)


def _periods(period_id: np.ndarray) -> list[np.ndarray]:
    """Frame indices at 5 Hz for each positive period."""
    return [np.flatnonzero(period_id == p)[::STRIDE] for p in np.unique(period_id[period_id > 0])]


class _Network:
    def __new__(cls, n_features: int, hidden: int, layers: int, dropout: float):
        torch = _torch()

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = torch.nn.Sequential(torch.nn.Linear(n_features, hidden), torch.nn.GELU())
                self.gru = torch.nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                                        bidirectional=True, dropout=dropout if layers > 1 else 0.0)
                self.out = torch.nn.Linear(2 * hidden, 1)

            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.out(h).squeeze(-1)

        return Network()


class SequenceHead:
    """One binary head. ``fit`` takes per-match arrays at the original frame rate."""

    def __init__(self, parameters: dict | None = None, device: str | None = None):
        self.parameters = {**SEQUENCE_PARAMETERS, **(parameters or {})}
        torch = _torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @_cpu_reproducible
    def fit(self, features: Sequence[np.ndarray], targets: Sequence[np.ndarray],
            masks: Sequence[np.ndarray], periods: Sequence[np.ndarray], fps: float,
            weights: Sequence[np.ndarray] | None = None):
        """``weights`` (per frame, aligned to ``features``) multiply the loss mask; the
        batch loss is then normalised by the weight sum instead of the frame count."""
        torch = _torch()
        p = self.parameters
        self.environment = {"device": str(self.device), "torch": str(torch.__version__),
                            "numpy": np.__version__, "python": platform.python_version(),
                            "platform": platform.platform(), "machine": platform.machine(),
                            "threads": torch.get_num_threads(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "cuda": torch.version.cuda,
                            "reproducibility": "same environment CPU; no cross-hardware bit identity guarantee"}
        rng = np.random.default_rng(p["seed"])
        torch.manual_seed(p["seed"])
        stacked = np.concatenate([x[m] for x, m in zip(features, masks)])
        self.centre = np.nanmean(stacked, axis=0)
        self.scale = np.nanstd(stacked, axis=0)
        self.centre = np.where(np.isfinite(self.centre), self.centre, 0.0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-6), self.scale, 1.0)
        # Only the statistics are needed below; do not retain the full masked
        # training copy during sequence encoding and thousands of GPU steps.
        del stacked
        sequences = []
        if weights is not None and len(weights) != len(features):
            raise ValueError("sequence weights must align to every training input")
        for k, (x, y, m, period_id) in enumerate(zip(features, targets, masks, periods)):
            w = None if weights is None else np.asarray(weights[k], dtype=np.float32)
            for idx in _periods(period_id):
                if m[idx].any():
                    loss_mask = (m[idx].astype(np.float32) if w is None
                                 else np.where(m[idx], w[idx], 0).astype(np.float32))
                    sequences.append((_encode(x[idx], self.centre, self.scale),
                                      np.where(m[idx], y[idx], 0).astype(np.float32), loss_mask))
        if not sequences:
            raise ValueError("no known frames to train a sequence head")
        window = max(2, int(p["window_s"] * fps / STRIDE))
        lengths = np.array([len(s[0]) for s in sequences], dtype=float)
        self.model = _Network(sequences[0][0].shape[1], p["hidden"], p["layers"], p["dropout"]).to(self.device)
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=p["learning_rate"], total_steps=p["steps"])
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.model.train()
        for _ in range(p["steps"]):
            picks = rng.choice(len(sequences), size=p["batch"], p=lengths / lengths.sum())
            batch = [self._crop(sequences[k], window, rng) for k in picks]
            x, y, m = (torch.from_numpy(np.stack(part)).to(self.device) for part in zip(*batch))
            # Unweighted masks sum to 0 or >= 1; weights (>= 0.25 per known frame) may sum below 1.
            loss = (loss_fn(self.model(x), y) * m).sum() / m.sum().clamp(min=1 if weights is None else 1e-6)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
        self.window = window
        return self

    def save(self, path) -> None:
        """Weights plus the training-only standardisation, enough to predict elsewhere."""
        torch = _torch()
        torch.save({"state": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    "centre": self.centre, "scale": self.scale, "window": self.window,
                    "parameters": self.parameters, "n_features": int(self.centre.size),
                    "environment": self.environment}, path)

    @classmethod
    def load(cls, path, device: str = "cpu", *, trusted_sha256: str) -> "SequenceHead":
        """Load only bytes matching an independently trusted checkpoint digest.

        The digest pins a trusted artifact; it is not a pickle scanner or sandbox.
        """
        checkpoint = Path(path).read_bytes()
        if hashlib.sha256(checkpoint).hexdigest() != trusted_sha256.lower():
            raise ValueError("checkpoint SHA256 does not match trusted digest")
        torch = _torch()
        blob = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)
        head = cls(blob["parameters"], device=device)
        head.centre, head.scale, head.window = blob["centre"], blob["scale"], blob["window"]
        head.environment = blob.get("environment", {"reproducibility": "training environment not recorded"})
        p = head.parameters
        head.model = _Network(2 * blob["n_features"], p["hidden"], p["layers"], p["dropout"]).to(device)
        head.model.load_state_dict(blob["state"])
        return head

    @staticmethod
    def _crop(sequence, window, rng):
        x, y, m = sequence
        if len(x) <= window:
            pad = window - len(x)
            return (np.pad(x, ((0, pad), (0, 0))), np.pad(y, (0, pad)), np.pad(m, (0, pad)))
        start = int(rng.integers(0, len(x) - window + 1))
        return x[start:start + window], y[start:start + window], m[start:start + window]

    @_cpu_reproducible
    def predict(self, features: np.ndarray, period_id: np.ndarray) -> np.ndarray:
        """Probability per frame; frames outside a period get 0.5 (they are never scored)."""
        torch = _torch()
        out = np.full(features.shape[0], 0.5)
        self.model.eval()
        overlap = int(self.parameters.get("window_overlap", 2))
        if overlap != 2 or self.parameters.get("triangular_average", False):
            return self._predict_overlapping(features, period_id, overlap,
                                             bool(self.parameters.get("triangular_average", False)))
        half = self.window // 2
        with torch.no_grad():
            for idx in _periods(period_id):
                x = _encode(features[idx], self.centre, self.scale)
                total, count = np.zeros(len(x)), np.zeros(len(x))
                starts = range(0, max(1, len(x) - half), half) if len(x) > self.window else [0]
                for start in starts:
                    piece = x[start:start + self.window]
                    logit = self.model(torch.from_numpy(piece[None]).to(self.device))[0].cpu().numpy()
                    total[start:start + len(piece)] += logit
                    count[start:start + len(piece)] += 1
                probability = 1 / (1 + np.exp(-total / np.maximum(count, 1)))
                frames = np.flatnonzero(period_id == period_id[idx[0]])
                out[frames] = np.interp(frames, idx, probability)
        return out

    def _predict_overlapping(self, features: np.ndarray, period_id: np.ndarray, overlap: int,
                             triangular: bool) -> np.ndarray:
        """Windows every ``window // overlap`` steps, the last one aligned to the period end; logits
        averaged with triangular weights (low at the window edges, where the BiGRU has little
        context) or uniformly. Selected by the head's ``window_overlap``/``triangular_average``."""
        if overlap < 1:
            raise ValueError("window_overlap must be a positive integer")
        torch = _torch()
        out = np.full(features.shape[0], 0.5)
        hop = max(1, self.window // overlap)
        with torch.no_grad():
            for idx in _periods(period_id):
                x = _encode(features[idx], self.centre, self.scale)
                n = len(x)
                if n > self.window:
                    starts = list(range(0, n - self.window + 1, hop))
                    if starts[-1] != n - self.window:
                        starts.append(n - self.window)
                else:
                    starts = [0]
                total, weight = np.zeros(n), np.zeros(n)
                for start in starts:
                    piece = x[start:start + self.window]
                    length = len(piece)
                    w = (np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1)).astype(np.float64)
                         if triangular else np.ones(length))
                    logit = self.model(torch.from_numpy(piece[None]).to(self.device))[0].cpu().numpy()
                    total[start:start + length] += w * logit
                    weight[start:start + length] += w
                probability = 1 / (1 + np.exp(-total / np.maximum(weight, 1e-9)))
                frames = np.flatnonzero(period_id == period_id[idx[0]])
                out[frames] = np.interp(frames, idx, probability)
        return out
