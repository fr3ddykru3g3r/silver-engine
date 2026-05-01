# PFIA v11 reproducibility materials

This folder contains the clean v11 reproducibility materials for the sports-technology manuscript.

## Contents

- `data/`: processed CSV files used in the manuscript.
- `code/`: reproduction script for the reported marathon and ATP summary outputs.
- `svgs/`: editable vector sources for equipment-interface figures.
- `requirements.txt`: minimal Python dependencies.

## Reproduction

```bash
pip install -r pfia-v11/requirements.txt
python pfia-v11/code/pfia_v11_reproduce.py
```

The script recomputes the marathon linear and segmented models, reports a Durbin-Watson diagnostic, and prints ATP endpoint and regression summaries.

The repository stores processed summary files rather than the full raw ATP match archive. The ATP source is Jeff Sackmann's public `tennis_atp` repository.
