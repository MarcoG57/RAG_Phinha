"""Strict, no-overwrite CSV export for the team's C4 submission."""
import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


_LABELS = {"H", "A", "D", "X"}


def _validated_rows(predictions, expected_frames):
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    try:
        iterator = iter(predictions)
    except TypeError as exc:
        raise ValueError("predictions must be iterable") from exc

    rows = []
    for expected_frame in range(expected_frames):
        try:
            prediction = next(iterator)
        except StopIteration as exc:
            raise ValueError("prediction count does not match expected_frames") from exc
        if not isinstance(prediction, Mapping):
            raise ValueError(f"prediction {expected_frame} must be a mapping")
        frame = prediction.get("frame")
        label = prediction.get("label")
        if type(frame) is not int or frame != expected_frame:
            raise ValueError("frames must be ordered, unique, and consecutive from zero")
        if not isinstance(label, str) or label not in _LABELS:
            raise ValueError(f"prediction {expected_frame} has a noncanonical label")
        rows.append((frame, label))
    try:
        next(iterator)
    except StopIteration:
        return rows
    raise ValueError("prediction count does not match expected_frames")


def write_submission(predictions, path, expected_frames):
    """Write canonical UTF-8/LF CSV once and return its SHA-256 hex digest."""
    rows = _validated_rows(predictions, expected_frames)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output:
            header = b"frame,label\n"
            output.write(header)
            digest.update(header)
            for frame, label in rows:
                line = f"{frame},{label}\n".encode("utf-8")
                output.write(line)
                digest.update(line)
            output.flush()
            os.fsync(output.fileno())

        # Windows rename refuses existing destinations; POSIX rename would replace one.
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
        return digest.hexdigest()
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
