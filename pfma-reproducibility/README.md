# PFMA reproducibility materials

This folder contains author-blind reproducibility materials for a sports-technology manuscript using Performance Frontier Mediation Analysis.

## Contents

- `data/`: processed CSV files used for the manuscript's marathon models and ATP annual aggregate summaries.
- `code/`: reproduction script for the reported marathon and ATP summary outputs.
- `svgs/`: editable vector figure sources for the equipment-interface diagrams.
- `requirements.txt`: minimal Python dependencies.

## Reproduction

From the repository root, install dependencies and run:

```bash
pip install -r pfma-reproducibility/requirements.txt
python pfma-reproducibility/code/pfma_v9_reproduce.py
```

The repository intentionally stores processed summary files rather than the full raw ATP match archive. The ATP source is Jeff Sackmann's public `tennis_atp` repository; the manuscript's annual aggregate outputs and model summaries are included here.

## Blind-review note

The files in this folder avoid author name, school name, local machine paths, Word metadata, and internal draft comments. For blind submission, link directly to this folder only if the journal permits repository usernames to be visible; otherwise archive the folder anonymously on OSF or Zenodo.
