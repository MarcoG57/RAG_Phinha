"""Audit external acquisitions before dense loading or runner-generated variants.

This is an integrity/provenance audit, not a sporting evaluation or a data licence
validator. Missing historical metadata is reported, never silently invented.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from possession.data.external import _local_path, _read_manifest, file_fingerprint

_FILES = {"synthetic": ("path",), "sportec": ("metadata", "positions"),
          "skillcorner": ("match", "tracking", "phases"), "skillcorner_legacy": ("match", "tracking")}
_OBSERVATIONS = {"synthetic": "path", "sportec": "positions", "skillcorner": "tracking",
                 "skillcorner_legacy": "tracking"}


def _files(fmt: str, row: dict) -> tuple[tuple[str, ...], str]:
    if fmt not in _FILES:
        raise ValueError("supported formats are synthetic, sportec, skillcorner, skillcorner_legacy")
    return _FILES[fmt], _OBSERVATIONS[fmt]


def _source_identity(fmt: str, paths: dict[str, Path], match_id: str) -> str | None:
    """Read provider identity from its owning metadata, independently of local names."""
    if fmt == "synthetic":
        return None  # synthetic fixtures have no real-world match identity
    if fmt in ("skillcorner", "skillcorner_legacy"):
        source_id = json.loads(paths["match"].read_text(encoding="utf-8")).get("id")
        if type(source_id) is not int or source_id < 0:
            raise ValueError("SkillCorner metadata needs a nonnegative integer source match id")
        expected_id = f"{'SC' if fmt == 'skillcorner' else 'SCL'}-{source_id}"
    else:
        try:
            general = ET.parse(paths["metadata"]).getroot().find(".//MatchInformation/General")
        except ET.ParseError as error:
            raise ValueError(f"malformed Sportec metadata: {error}") from error
        source_id = None if general is None else general.get("MatchId")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Sportec metadata needs a source match id")
        expected_id = source_id
    if match_id != expected_id:
        raise ValueError(f"manifest id {match_id!r} disagrees with {fmt} source match id {source_id!r}")
    return str(source_id)


def audit_manifests(manifest_paths: Iterable[Path]) -> dict:
    """Validate acquisition identity across all primary and extra manifests.

    Raises ``ValueError`` for repeated global ids, repeated provider identities,
    byte-identical observation acquisitions or metadata/manifest id disagreement.
    Source/version/license are optional declarations; absence is recorded. The
    report contains counts and SHA-256 fingerprints, no arrays or absolute paths.

    Call once before loading/training. Planned detected/extrapolated, scenario or
    seed variants are derived afterwards from these acquisitions, keeping their
    original ``match_id`` for grouped splitting. Repeating a manifest row to create
    a variant is an acquisition duplicate and is deliberately rejected.

    Hashing streams the input bytes once; parsing provider metadata is small.
    It detects byte-identical reuse, not arbitrary re-encoding across providers.
    """
    manifests = [_read_manifest(path) for path in manifest_paths]
    if not manifests:
        raise ValueError("at least one external manifest is required")
    ids = [row["id"] for _, _, rows in manifests for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate match id across external manifests")
    identities, observations, rows_out = {}, {}, []
    counts = Counter()
    for manifest_path, manifest, rows in manifests:
        for row in rows:
            match_id, fmt = row["id"], row["format"]
            files, observation_key = _files(fmt, row)
            paths = {key: _local_path(manifest_path, row, key) for key in files}
            source_id = _source_identity(fmt, paths, match_id)
            # Both SkillCorner schemas use the provider match id namespace.
            provider = "skillcorner" if fmt == "skillcorner_legacy" else fmt
            identity = (provider, source_id)
            if source_id is not None:
                if identity in identities:
                    raise ValueError(f"duplicate physical match identity: {match_id} and {identities[identity]}")
                identities[identity] = match_id
            fingerprints = {}
            for key, path in paths.items():
                fingerprint = file_fingerprint(path)
                fingerprints[key] = {"sha256": fingerprint["sha256"], "size_bytes": fingerprint["size_bytes"]}
            observation_hash = fingerprints[observation_key]["sha256"]
            if observation_hash in observations:
                raise ValueError(f"duplicate observation acquisition: {match_id} and {observations[observation_hash]}")
            observations[observation_hash] = match_id
            declared = {**manifest.get("provenance", {}), **row.get("provenance", {})}
            metadata = {key: declared.get(key, declared.get("source_version") if key == "version" else None)
                        for key in ("source", "version", "license")}
            # TOML dates are legal declarations; report them as ISO strings for JSON receipts.
            metadata = json.loads(json.dumps(metadata, default=str))
            rows_out.append({"match_id": match_id, "format": fmt, "provider": fmt,
                             "source_match_id": source_id, "fingerprints": fingerprints,
                             "metadata": metadata,
                             "missing_metadata": [key for key, value in metadata.items() if value in (None, "")]})
            counts[fmt] += 1
    return {"schema_version": 1, "manifest_count": len(manifests), "match_count": len(rows_out),
            "counts_by_format": dict(sorted(counts.items())), "matches": rows_out}
