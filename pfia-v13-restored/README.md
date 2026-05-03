# PFIA v19 FINALPOLISH submission package

This package is the polished PFIA submission package after the v18 METHODFIX audit.

Implemented since v18:
- versioned `requirements.txt` added/expanded;
- `CHANGELOG.md` added;
- bootstrap-interval documentation note added to the manuscript;
- Figure 5 label/legend crowding fixed;
- Figure 6 label clipping fixed with a cleaner vertical chart;
- Table 3 row spillover fixed;
- manuscript rendered and visually checked.

Core v18 upgrades retained:
- pre-2018-only smooth-progression top-list null;
- numbered equations;
- rival-framework positioning;
- three propositional PFIA predictions;
- corrected figure manifest;
- editable SVGs for all figures;
- reproduction scripts and logs.

Run from package root:

```bash
pip install -r requirements.txt
python code/pfia_v18_reproduce.py
```

High-effort future work still requires new external data: annual top-50/top-100 marathon distributions, player-level ATP mixed-effects models, women/WTA replication, Bayesian breakpoint modelling, and football empirical data.

Do not submit older PFMA files, v15 visual-final files, or any package containing internal review/change-log language.