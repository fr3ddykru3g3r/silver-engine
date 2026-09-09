#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from flare_system.aarp import (
    M1_24H_BYTES,
    M1_24H_URL,
    audit_parameter_file,
    verify_parameter_file,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def download_verified(destination: Path, timeout: int = 180) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        return verify_parameter_file(destination)
    except RuntimeError:
        pass
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(M1_24H_URL, headers={"User-Agent": "IRIS-AARP-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != M1_24H_BYTES:
                raise RuntimeError(
                    f"AARP_DOWNLOAD_SIZE_HEADER_INVALID: {content_length} != {M1_24H_BYTES}"
                )
            shutil.copyfileobj(response, output, length=1 << 20)
        verification = verify_parameter_file(temporary)
        os.replace(temporary, destination)
        return verification
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and audit the Harvard AARP flare parameter file")
    parser.add_argument("--parameter-file", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    parameter_file = Path(args.parameter_file)
    receipt = Path(args.receipt)
    try:
        if args.download:
            download_verified(parameter_file, args.timeout)
        result = audit_parameter_file(parameter_file)
        atomic_json(receipt, result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        failure = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(receipt, failure)
        print(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
