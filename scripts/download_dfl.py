"""Explicit local acquisition of the public DFL dataset from Figshare; no implicit downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.figshare.com/v2/articles/28196177/versions/1"
DOI = "10.6084/m9.figshare.28196177.v1"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MAX_INVENTORY_BYTES = 1024 * 1024
MAX_FILE_BYTES = 2_000_000_000
CHUNK_BYTES = 1024 * 1024
TIMEOUT_S = 60
NAME = re.compile(r"DFL_(02_01_matchinformation|04_03_positions_raw_observed)_DFL-COM-[0-9]{6}_(DFL-MAT-[A-Z0-9]{6})\.xml\Z")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        # The published ndownloader endpoint redirects to Figshare's public S3 storage.
        if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
                or parsed.hostname not in {"ndownloader.figshare.com", "s3-eu-west-1.amazonaws.com", "s3.amazonaws.com"}):
            raise ValueError("unapproved Figshare download redirect")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _open(url):
    return urllib.request.build_opener(_SafeRedirect()).open(url, timeout=TIMEOUT_S)


def fetch_inventory():
    with _open(API_URL) as response:
        raw = response.read(MAX_INVENTORY_BYTES + 1)
    if len(raw) > MAX_INVENTORY_BYTES:
        raise ValueError("Figshare inventory exceeds the 1 MiB bound")
    return json.loads(raw)


def inventory_pairs(inventory):
    if (type(inventory.get("id")) is not int or inventory["id"] != 28196177
            or type(inventory.get("version")) is not int or inventory["version"] != 1
            or inventory.get("doi") != DOI or inventory.get("license", {}).get("url") != LICENSE_URL):
        raise ValueError("expected Figshare article 28196177 version 1 under CC BY 4.0")
    pairs = {}
    for entry in inventory.get("files", []):
        name = entry.get("name", "")
        if not isinstance(name, str) or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("unsafe name in published inventory")
        match = NAME.fullmatch(name)
        if match is None:
            continue  # Event files are outside this acquisition surface.
        identifier, size = entry.get("id"), entry.get("size")
        if type(identifier) is not int or identifier <= 0 or type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError("invalid file id or bounded size in inventory")
        if entry.get("is_link_only") or entry.get("download_url") != f"https://ndownloader.figshare.com/files/{identifier}":
            raise ValueError("only published HTTPS ndownloader file URLs are allowed")
        hashes = [str(entry[key]).lower() for key in ("computed_md5", "supplied_md5") if entry.get(key)]
        if any(not re.fullmatch(r"[0-9a-f]{32}", value) for value in hashes) or len(set(hashes)) > 1:
            raise ValueError("invalid or conflicting published MD5")
        kind = "metadata" if match[1] == "02_01_matchinformation" else "positions"
        pair = pairs.setdefault(match[2], {})
        if kind in pair:
            raise ValueError("duplicate published match/file kind")
        pair[kind] = {"id": identifier, "name": name, "size": size,
                      "download_url": entry["download_url"], "md5": hashes[0] if hashes else None}
    if not pairs or any(set(pair) != {"metadata", "positions"} for pair in pairs.values()):
        raise ValueError("inventory needs complete metadata/positions pairs")
    return pairs


def _verified(path, entry):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        return False
    if entry["md5"] is None:
        return True
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest() == entry["md5"]


def _publish(part, target):
    # Windows rename refuses an existing target. POSIX rename would overwrite it,
    # so publish there by exclusive hard-link creation, then unlink the temporary.
    # Adjacent paths keep publication on the same filesystem.
    if os.name == "nt":
        os.rename(part, target)
    else:
        os.link(part, target)
        part.unlink()


def download_file(entry, target):
    target = Path(target)
    if target.exists() or target.is_symlink():
        if _verified(target, entry):
            return "reused"
        raise ValueError("existing destination fails published size/MD5 validation; preserved")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    created = False
    try:
        with part.open("xb") as handle:
            created = True
            total = 0
            digest = hashlib.md5()
            with _open(entry["download_url"]) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != entry["size"]:
                    raise ValueError("HTTP Content-Length disagrees with inventory")
                while True:
                    chunk = response.read(min(CHUNK_BYTES, entry["size"] - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > entry["size"]:
                        raise ValueError("download exceeds published size")
                    handle.write(chunk)
                    digest.update(chunk)
            if total != entry["size"] or (entry["md5"] and digest.hexdigest() != entry["md5"]):
                raise ValueError("download fails published size/MD5 validation")
            handle.flush()
            os.fsync(handle.fileno())
        _publish(part, target)
        return "downloaded"
    finally:
        if created and part.exists():
            part.unlink()  # Only this invocation's incomplete temporary file.


def _text_preflight(path, text):
    if path.is_symlink() or (path.exists() and path.read_text(encoding="utf-8") != text):
        raise ValueError("existing manifest differs; choose another output directory")


def _write_new_text(path, text):
    _text_preflight(path, text)
    if path.exists():
        return
    part = path.with_name(path.name + ".part")
    with part.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    try:
        _publish(part, path)
    finally:
        if part.exists():
            part.unlink()


def acquire(out, matches, *, accept_license):
    if accept_license != "CC-BY4":
        raise ValueError("explicit --accept-license CC-BY4 is required before network access")
    out = Path(out).resolve()
    if not any(out.is_relative_to(ROOT.resolve() / area) for area in ("data", "artifacts")):
        raise ValueError("--out must resolve inside this repository's data/ or artifacts/")
    inventory = fetch_inventory()
    pairs = inventory_pairs(inventory)
    selected = sorted(pairs if matches is None else set(matches))
    if not selected or any(match not in pairs for match in selected):
        raise ValueError("requested match not present in published inventory")
    lines = ["# Public DFL inputs; acquisition does not validate parser or sporting performance.",
             "[provenance]", f'source = "{API_URL}"', f'license = "{LICENSE_URL}"',
             f'doi = {json.dumps(inventory.get("doi", ""))}', ""]
    for match in selected:
        folder = match.removeprefix("DFL-MAT-")
        lines += ["[[matches]]", f'id = "{match}"', 'format = "sportec"',
                  f'metadata = "{folder}/metadata.xml"', f'positions = "{folder}/positions.xml"',
                  "fps = 25", 'coordinate_origin = "center"', 'coordinate_units = "m"', 'speed_units = "km/h"', ""]
    manifest_text = "\n".join(lines)
    receipt = {"schema_version": 1, "article_id": 28196177, "source": API_URL, "license": LICENSE_URL,
               "doi": inventory.get("doi"), "citation": "Bassek et al. (2025), Scientific Data 12, 195; doi:10.1038/s41597-025-04505-y",
               "acquisition_only": True, "matches": {match: pairs[match] for match in selected}}
    receipt_text = json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    _text_preflight(out / "manifest.toml", manifest_text)
    _text_preflight(out / "download-manifest.json", receipt_text)
    for match in selected:
        folder = out / match.removeprefix("DFL-MAT-")
        if not folder.resolve().is_relative_to(out):
            raise ValueError("match directory escapes --out")
        for kind in ("metadata", "positions"):
            state = download_file(pairs[match][kind], folder / f"{kind}.xml")
            print(f"{state} {match} {kind}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    _write_new_text(out / "manifest.toml", manifest_text)
    _write_new_text(out / "download-manifest.json", receipt_text)
    return out / "manifest.toml"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--matches", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--accept-license", choices=["CC-BY4"], required=True)
    args = parser.parse_args(argv)
    try:
        path = acquire(args.out, None if args.all else args.matches, accept_license=args.accept_license)
        print(f"PASS acquisition; evaluation manifest: {path.name}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as error:
        print(f"FAIL acquisition: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
