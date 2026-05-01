# PFIA v11

Run:

```bash
pip install -r pfia-v11/requirements.txt
python pfia-v11/code/pfia_v11_reproduce.py
```

The script reads processed CSV files from `pfia-v11/data/` when present. If that folder is absent, it reads the existing files in `pfma-reproducibility/data/`.

The script recomputes the marathon models, reports a Durbin-Watson diagnostic, and prints ATP endpoint and regression summaries.
