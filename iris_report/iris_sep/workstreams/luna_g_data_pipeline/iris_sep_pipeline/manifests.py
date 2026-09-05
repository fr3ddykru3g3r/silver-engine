"""Canonical immutable manifest helpers for synthetic contract testing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .errors import PipelineError


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def freeze_manifest(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not kind or not isinstance(payload, Mapping):
        raise PipelineError("manifest kind and mapping payload are required")
    core = {"schema_version": 1, "kind": kind, "status": "FROZEN", "payload": dict(payload)}
    return {**core, "manifest_sha256": sha256_canonical(core)}


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    required = {"schema_version", "kind", "status", "payload", "manifest_sha256"}
    if set(manifest) != required or manifest.get("status") != "FROZEN":
        return False
    core = {name: manifest[name] for name in ("schema_version", "kind", "status", "payload")}
    return manifest["manifest_sha256"] == sha256_canonical(core)


def write_immutable_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not verify_manifest(manifest):
        raise PipelineError("refusing to write an invalid manifest")
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != dict(manifest):
            raise PipelineError("immutable manifest already exists with different content")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(dict(manifest)) + b"\n")
    return destination
