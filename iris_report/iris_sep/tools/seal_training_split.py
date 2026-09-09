"""Create a development-only V2 table from an authoritative training allowlist.

The tool emits no row identities or label values. Matching is exact on the full
`(window_begin, window_end)` strings. Excluded source rows are never written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


class SealError(ValueError):
    pass


KEYS = ("window_begin", "window_end")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_allowlist(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(KEYS).issubset(reader.fieldnames):
            raise SealError("allowlist is missing exact window keys")
        result: set[tuple[str, str]] = set()
        for row in reader:
            key = tuple(row[name] for name in KEYS)
            if not all(key) or key in result:
                raise SealError("allowlist contains empty or duplicate exact keys")
            result.add(key)
    if not result:
        raise SealError("allowlist is empty")
    return result


def seal_training_rows(source: Path, allowlist_path: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        raise SealError("destination already exists; sealed artifacts are immutable")
    allowed = read_allowlist(allowlist_path)
    matched: set[tuple[str, str]] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    source_rows = 0
    try:
        with source.open(newline="", encoding="utf-8-sig") as input_stream:
            reader = csv.DictReader(input_stream)
            if reader.fieldnames is None or not set(KEYS).issubset(reader.fieldnames):
                raise SealError("source is missing exact window keys")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise SealError("source has duplicate columns")
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", delete=False,
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
            ) as output_stream:
                temporary = Path(output_stream.name)
                writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    source_rows += 1
                    key = tuple(row[name] for name in KEYS)
                    if key in allowed:
                        if key in matched:
                            raise SealError("source contains duplicate allowed key")
                        writer.writerow(row)
                        matched.add(key)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        if matched != allowed:
            raise SealError(
                f"exact-key coverage failure: matched {len(matched)} of {len(allowed)} training keys"
            )
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "status": "PASS",
        "matching": "EXACT_WINDOW_BEGIN_AND_END_ONLY",
        "source_sha256": sha256_file(source),
        "allowlist_sha256": sha256_file(allowlist_path),
        "output_sha256": sha256_file(destination),
        "source_rows": source_rows,
        "allowlist_rows": len(allowed),
        "matched_rows": len(matched),
        "excluded_rows_not_written": source_rows - len(matched),
        "excluded_identities_or_outcomes_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise SealError("receipt already exists; receipts are immutable")
    receipt = seal_training_rows(args.source, args.allowlist, args.output)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
