"""Hashes and constants that tie training reports and model registrations to dataset versions."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

REPORT_SCHEMA_VERSION = 2
SPLIT_SEED = 42


def id_fingerprint(ids: Iterable[object]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
