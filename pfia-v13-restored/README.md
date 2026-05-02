# PFIA v13 restored submission package

This folder mirrors the restored PFIA v13 submission package.

Canonical manuscript title:

**Performance Frontier Interface Analysis (PFIA): A Cross-Sport Framework for Running Shoes, Tennis Rackets, and Football Boots in Elite Performance**

This restored package uses the stronger verified v13 manuscript base and replaces the later regressed figure set with cleaner publication-safe figures. It should be treated as the current PFIA source of truth unless a later version explicitly supersedes it.

## Contents

- `code/`: reproduction script.
- `data/`: processed CSV files used by the script and manuscript tables.
- `svgs/`: editable SVG sources for all eleven figures.
- `figure_manifest.csv`: manuscript figure-number mapping.
- `repro_check.txt`: expected reproduction output.
- `final_text_audit.txt`: audit showing no internal/revision phrases.
- `requirements.txt`: Python dependencies.

The DOCX/PDF manuscript and rendered PNG figure files are included in the downloadable chat package. This GitHub mirror stores source/reproducibility files and SVG figure sources so the package remains inspectable in the repository.

## Run

```bash
pip install -r requirements.txt
python code/pfia_v13_reproduce.py
```

## Important

Do **not** submit older PFMA files, v15 visual-final files, or any package containing internal review/change-log language. This folder is the restored PFIA v13 source package.
