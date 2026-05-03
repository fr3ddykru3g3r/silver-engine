# PFIA v18 METHODFIX submission package

This package builds on PFIA v17 DEEPENED and addresses the next audit finding: the smooth-progression top-list null should not be fitted on the full data that contain the post-2017 concentration being tested.

Implemented upgrades:
- Fits the top-list progression-null simulation only on pre-2018 performances.
- Adds `toplist_pre2018_progression_null_v18.csv`.
- Adds `code/pfia_v18_reproduce.py`.
- Removes the duplicated top-list result from the marathon-frontier Results subsection.
- Moves marathon and tennis confound discussion into the Discussion section.
- Adds a clearer ceiling statement: the current paper still needs annual top-50/top-100 marathon data, player-level ATP mixed-effects models, women/WTA replication, and an empirical football module for a much stronger PFIA 2.0 version.
- Keeps corrected figures, editable SVG sources, v17 rival-framework positioning, and the propositional PFIA predictions.

Run:

```bash
pip install -r requirements.txt
python code/pfia_v18_reproduce.py
```

The high-effort upgrades requiring new external datasets remain future work, not fabricated additions. Do not submit older PFMA files, v15 visual-final files, or any package containing internal review/change-log language.