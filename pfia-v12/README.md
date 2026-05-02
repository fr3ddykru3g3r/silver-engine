# PFIA v12 submission-ready reproducibility materials

This folder tracks the corrected PFIA v12 submission package. It is **not** the older draft file and should not contain internal coaching/meta-commentary.

Use the manuscript titled:

`Performance Frontier Interface Analysis (PFIA): A Cross-Sport Framework for Running Shoes, Tennis Rackets, and Football Boots in Elite Performance`

Do **not** submit any file that says the ATP module was not executed, says “not peer reviewed,” uses the older PFMA draft framing, or contains AJSR-style coaching notes.

Run:

```bash
pip install -r pfia-v12/requirements.txt
python pfia-v12/code/pfia_v12_reproduce.py
```

The script recomputes the marathon frontier models, top-list compression intervals, ATP endpoint summaries, and ATP surface-interaction model used as the PFIA prediction check.

For formal journal submission, upload the full cleaned package as an anonymized supplement or archive it on OSF/Zenodo.