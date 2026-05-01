# PFMA reproducibility materials

This folder contains reproducibility materials for a sports-technology manuscript.

## Contents

- `data/`: processed summary CSV files used in the manuscript.
- `code/`: scripts to reproduce the reported marathon and ATP summary outputs.
- `svgs/`: editable vector figure sources.
- `figures/`: manuscript figure PNGs are available in the manuscript package.

## Reproduction

Run:

```bash
python code/pfma_v9_reproduce.py
```

The script recomputes the marathon linear and segmented models and prints the ATP endpoint-change table from the processed CSV files.
