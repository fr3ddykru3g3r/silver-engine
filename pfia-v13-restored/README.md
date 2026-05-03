# PFIA v16 PLUS / restored PFIA submission package

This folder mirrors the controlled PFIA repair package.

Canonical manuscript title:

**Performance Frontier Interface Analysis (PFIA): A Cross-Sport Framework for Running Shoes, Tennis Rackets, and Football Boots in Elite Performance**

This package uses the stronger verified PFIA v13 manuscript as the base and applies the final targeted reviewer fixes:

- numbered marathon equations (1) and (2)
- regenerated publication-safe figures with corrected margins and visible interface details
- synthesis paragraph connecting marathon, tennis, and football modules
- explicit marathon confounds paragraph
- AIC interpretation note for the small record-series model
- discussion justifying 2017 as a theoretical breakpoint while acknowledging 2014 as the minimum-AIC candidate
- explicit theoretical contribution statement
- ATP confound discussion
- future-research agenda paragraph
- editable SVG source files for all eleven figures

## Contents

- `code/`: reproduction script.
- `data/`: processed CSV files used by the script and manuscript tables.
- `svgs/`: editable SVG sources for all eleven figures.
- `figure_manifest.csv`: manuscript figure-number mapping.
- `repro_check.txt`: expected reproduction output.
- `final_text_audit.txt`: audit showing no internal/revision phrases.
- `requirements.txt`: Python dependencies.

The DOCX/PDF manuscript and rendered PNG figures are included in the downloadable chat package. This GitHub mirror stores source/reproducibility files and SVG figure sources so the package remains inspectable in the repository.

## Run

```bash
pip install -r requirements.txt
python code/pfia_v13_reproduce.py
```

The script filename remains `pfia_v13_reproduce.py` because the numerical package is still based on the verified v13 data/model outputs; the manuscript has been upgraded through controlled text and figure patches.

## Important

Do **not** submit older PFMA files, v15 visual-final files, or any package containing internal review/change-log language. This folder is the current PFIA source package.