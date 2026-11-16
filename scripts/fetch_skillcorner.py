"""Acquire the pinned public SkillCorner corpus; no credentials, dependencies or implicit download.

Without --download this only prints the pinned source and required file names.
With --download, existing files are preserved and LFS objects are checked against
their upstream SHA256 and size before publication. Partial transfers remain .part
and never become input files. No match from BAR-ATH is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}"
MEDIA = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{COMMIT}"
IDS = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
       2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461)


def lfs_identity(pointer: bytes):
    text = pointer.decode("ascii")
    digest = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
    size = re.search(r"^size ([0-9]+)$", text, re.M)
    if not text.startswith("version https://git-lfs.github.com/spec/v1\n") or not digest or not size:
        raise ValueError("upstream tracking file must be a canonical Git LFS pointer")
    return digest[1], int(size[1])


def identity(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest, Path(path).stat().st_size


def _publish(partial, path):
    # POSIX rename would overwrite a concurrent destination.
    if os.name == "nt":
        os.rename(partial, path)
    else:
        os.link(partial, path)
        partial.unlink()


def fetch(url, path, expected=None):
    path = Path(path)
    if path.exists():
        actual = identity(path)
        if expected is None:
            # Metadata have no LFS pointer: compare the small immutable response
            # too, so a resumed acquisition cannot bless an edited local file.
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=90) as response:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            expected = digest.hexdigest(), size
        if actual != expected:
            raise ValueError(f"existing input differs from pinned upstream: {path.name}")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    # Exclusive creation: never overwrite another transfer or silently reuse partial bytes.
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=90) as response, partial.open("xb") as stream:
        while block := response.read(1024 * 1024):
            digest.update(block)
            size += len(block)
            stream.write(block)
    actual = digest.hexdigest(), size
    if expected is not None and actual != expected:
        raise ValueError(f"download SHA256/size mismatch: {path.name}; .part retained for diagnosis")
    if path.exists():
        raise ValueError(f"input appeared during transfer: {path.name}")
    _publish(partial, path)
    return actual


def manifest_text(ids, players):
    rows = ["# Public SkillCorner inputs, pinned acquisition; no sporting-performance claim.",
            "[provenance]", f'source = "https://github.com/SkillCorner/opendata/tree/{COMMIT}"',
            f'version = "{COMMIT}"',
            f'license = "https://github.com/SkillCorner/opendata/blob/{COMMIT}/LICENSE"',
            'attribution = "SkillCorner Open Data / PySport"', ""]
    for mid in ids:
        rows += ["[[matches]]", f'id = "SC-{mid}"', 'format = "skillcorner"',
                 f'match = "{mid}/{mid}_match.json"', f'tracking = "{mid}/{mid}_tracking_extrapolated.jsonl"',
                 f'phases = "{mid}/{mid}_phases_of_play.csv"', f'players = "{players}"', ""]
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--ids", type=int, nargs="+", default=list(IDS))
    parser.add_argument("--players", choices=("detected", "extrapolated"), default="extrapolated")
    parser.add_argument("--out", type=Path, default=ROOT / "data/external/skillcorner")
    args = parser.parse_args(argv)
    if not args.ids or len(set(args.ids)) != len(args.ids) or not set(args.ids) <= set(IDS):
        raise ValueError("ids must be distinct members of the pinned official corpus")
    out = args.out.resolve()
    allowed = (ROOT / "data/external").resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("output must be a child of this checkout's data/external/")
    if not args.download:
        print(json.dumps({"source": RAW, "commit": COMMIT, "matches": args.ids,
                          "estimated_full_corpus_tracking_bytes": 1823448331,
                          "download": "NOT RUN; pass --download explicitly"}, indent=2))
        return 0
    for name in ("LICENSE", "README.md"):
        fetch(f"{RAW}/{name}", out / name)
    fetch(f"{RAW}/data/matches.json", out / "matches.json")
    manifest = json.loads((out / "matches.json").read_text(encoding="utf-8"))
    actual_ids = [int(row["id"]) for row in manifest]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(IDS):
        raise ValueError("upstream corpus disagrees with the pinned 20 unique ids")
    records = []
    for mid in args.ids:
        files = {}
        folder = out / str(mid)
        for suffix in ("match.json", "phases_of_play.csv"):
            name = f"{mid}_{suffix}"
            digest, size = fetch(f"{RAW}/data/matches/{mid}/{name}", folder / name)
            files[suffix] = {"sha256": digest, "size_bytes": size}
        name = f"{mid}_tracking_extrapolated.jsonl"
        with urllib.request.urlopen(f"{RAW}/data/matches/{mid}/{name}", timeout=90) as response:
            expected = lfs_identity(response.read(4096))
        digest, size = fetch(f"{MEDIA}/data/matches/{mid}/{name}", folder / name, expected)
        files["tracking"] = {"sha256": digest, "size_bytes": size}
        records.append({"id": mid, "files": files})
        print(f"{mid}: pinned tracking verified, {size} bytes", flush=True)
    for players in ("detected", "extrapolated"):
        path = out / f"manifest-{players}.toml"
        text = manifest_text(args.ids, players)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError("existing manifest differs; preserve it and choose a fresh output")
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    record = {"source_commit": COMMIT, "match_count": len(args.ids), "matches": records,
              "attribution": "SkillCorner Open Data / PySport", "default_players": args.players}
    (out / "acquisition.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(records)} public matches; {out / ('manifest-' + args.players + '.toml')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
