from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from iris_report.iris_sep.tools.seal_training_split import SealError, seal_training_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class SealTrainingSplitTests(unittest.TestCase):
    def test_only_exact_allowlisted_rows_are_written(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iris_sep_seal_") as directory:
            root = Path(directory); source = root / "source.csv"; allowed = root / "allowed.csv"; output = root / "output.csv"
            write_csv(source, [
                {"window_begin": "a", "window_end": "b", "Future_secret": "0"},
                {"window_begin": "c", "window_end": "d", "Future_secret": "1"},
            ])
            write_csv(allowed, [{"window_begin": "a", "window_end": "b", "target": "safe"}])
            receipt = seal_training_rows(source, allowed, output)
            self.assertEqual(receipt["matched_rows"], 1)
            self.assertEqual(receipt["excluded_rows_not_written"], 1)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(row["window_begin"], row["window_end"]) for row in rows], [("a", "b")])

    def test_near_or_partial_time_match_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iris_sep_seal_") as directory:
            root = Path(directory); source = root / "source.csv"; allowed = root / "allowed.csv"
            write_csv(source, [{"window_begin": "2026-01-01T00:00", "window_end": "2026-01-02T00:00", "x": "1"}])
            write_csv(allowed, [{"window_begin": "2026-01-01T00:01", "window_end": "2026-01-02T00:01"}])
            with self.assertRaisesRegex(SealError, "coverage failure"):
                seal_training_rows(source, allowed, root / "output.csv")


if __name__ == "__main__":
    unittest.main()
