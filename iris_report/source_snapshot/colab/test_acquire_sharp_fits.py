"""Offline tests for JSOC export record matching."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
from acquire_sharp_fits import normalize_trec, record_key, scope_flags


def main() -> None:
    assert normalize_trec(" 2010.05.01_00:05:59_TAI ") == "2010.05.01_00:05:59_TAI"
    assert record_key("hmi.sharp_cea_720s[401][2010.05.01_00:05:59_TAI]") == (
        401, "2010.05.01_00:05:59_TAI"
    )
    assert record_key("not-a-record") is None
    assert scope_flags("base") == (True, False, False)
    assert scope_flags("physics") == (False, True, False)
    assert scope_flags("downstream") == (False, False, True)
    assert scope_flags("all") == (True, True, True)
    print("acquisition helper self-test PASS")


if __name__ == "__main__":
    main()
