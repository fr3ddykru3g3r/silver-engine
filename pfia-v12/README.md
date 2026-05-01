# PFIA v12 reproducibility materials

This folder contains the v12 reproduction script for the PFIA sports-technology manuscript.

Run:

```bash
pip install -r pfia-v12/requirements.txt
python pfia-v12/code/pfia_v12_reproduce.py
```

The script reads processed CSV files from `pfia-v12/data/` if present. If that folder is absent, it uses the existing processed data in `pfma-reproducibility/data/`.

It recomputes the marathon frontier models, top-list compression intervals, ATP endpoint summaries, and the ATP surface-interaction model used as the PFIA prediction check.
