"""Reconstruct and execute the frozen freshness-crossover V1 study source.

The scientific implementation is stored as fixed base64 chunks only because
GitHub connector writes are UTF-8 text operations. This launcher checks the
exact decoded SHA-256 before importing or executing any study code, so a
transport/copy error fails before acquisition, label derivation, or scoring.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import sys

EXPECTED_SOURCE_SHA256 = "7eb98fa0f3fb8bb1d01270cc70ebd895ac4822f90f83e811f6d1d112e1c74971"
PAYLOAD_DIR = Path(__file__).resolve().parent / "freshness_crossover_v1_payload"


def reconstruct() -> bytes:
    parts = sorted(PAYLOAD_DIR.glob("part*.b64"))
    if [p.name for p in parts] != [f"part{i:02d}.b64" for i in range(1, 6)]:
        raise SystemExit(f"Expected exactly part01..part05, found {[p.name for p in parts]}")
    encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise SystemExit(f"Payload base64 validation failed: {exc}") from exc
    observed = hashlib.sha256(raw).hexdigest()
    if observed != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "Decoded source SHA-256 mismatch before study execution: "
            f"expected={EXPECTED_SOURCE_SHA256} observed={observed}"
        )
    return raw


def main() -> int:
    raw = reconstruct()
    decoded = Path("/tmp/run_freshness_crossover_study_v1.py")
    decoded.write_bytes(raw)
    code = compile(raw, str(decoded), "exec")
    namespace = {"__name__": "__main__", "__file__": str(decoded)}
    # Preserve command-line arguments such as --output for the reconstructed
    # script. Any SystemExit from the study is intentionally propagated.
    exec(code, namespace, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
