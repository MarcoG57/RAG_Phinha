"""Locate the pinned, unmodified public starter; never download it implicitly."""
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def challenge_config() -> dict:
    return tomllib.loads((ROOT / "configs" / "challenge.toml").read_text(encoding="utf-8"))


def checked_official() -> Path:
    official = ROOT / ".official"
    command = ["git", "-C", str(official)]
    head = subprocess.run(command + ["rev-parse", "HEAD"], capture_output=True, text=True)
    expected = challenge_config()["upstream"]["commit"]
    if head.returncode or head.stdout.strip() != expected:
        raise ValueError("official checkout missing or wrong commit; follow README.md")
    # Scoring output lives in artifacts/, so tracked source should remain unchanged.
    clean = subprocess.run(command + ["diff", "--quiet", "HEAD", "--"], capture_output=True)
    if clean.returncode:
        raise ValueError("official tracked files modified; restore/reference deliberately")
    return official
