"""Strict team CSV preflight followed by the unchanged official validator."""
import argparse
import csv
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from official_checkout import ROOT, checked_official, challenge_config


def iter_submission_labels(path: Path, n_expected: int | None) -> Iterator[str]:
    """Yield canonical rows; None skips the fixed count for the public GT."""
    if n_expected is not None and n_expected <= 0:
        raise ValueError("expected count must be positive")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, strict=True)
        if next(rows, None) != ["frame", "label"]:
            raise ValueError("header must be exactly frame,label")
        count = 0
        for count, row in enumerate(rows, start=1):
            if n_expected is not None and count > n_expected:
                raise ValueError(f"more than {n_expected} rows")
            if len(row) != 2:
                raise ValueError(f"row {count}: expected two cells")
            if row[0] != str(count - 1):
                raise ValueError(f"row {count}: frame must be {count - 1}")
            if row[1] not in {"H", "A", "D", "X"}:
                raise ValueError(f"row {count}: expected canonical H/A/D/X label")
            yield row[1]
        if n_expected is not None and count != n_expected:
            raise ValueError(f"row count {count} != {n_expected}")


def read_submission_labels(path: Path, n_expected: int | None = None) -> list[str]:
    """Read one C4 submission in one pass; default to the challenge frame count."""
    if n_expected is None:
        n_expected = challenge_config()["challenge"]["frames"]
    return list(iter_submission_labels(path, n_expected))


def strict_preflight(path: Path, n_expected: int) -> None:
    """Reject any violation of the team's canonical C4 export contract in O(1) memory."""
    if n_expected is None:
        raise ValueError("expected count must be positive")
    for _ in iter_submission_labels(path, n_expected):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="CSV file to check")
    args = parser.parse_args()
    try:
        expected = challenge_config()["challenge"]["frames"]
        path = args.path.resolve()
        strict_preflight(path, expected)
        official = checked_official()
    except (OSError, ValueError, csv.Error) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    print(f"PASS strict C4 preflight ({expected} rows)", flush=True)
    return subprocess.run([
        sys.executable, str(official / "evaluation" / "validate_submission.py"),
        str(path), "--rows", str(expected),
    ], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
