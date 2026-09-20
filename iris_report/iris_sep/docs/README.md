# Research-reference paper

- Source: [`sep_evaluation_research_reference.tex`](sep_evaluation_research_reference.tex)
- Compiled PDF: [`build/sep_evaluation_research_reference.pdf`](build/sep_evaluation_research_reference.pdf)

This is an AI-authored internal reference built from the audited evidence package. It explains the findings, equations, usefulness, limitations, negative results, falsification conditions, and competition assessment in research-paper form. It is not student-authored initial submission prose.

Compile from the LaTeX plugin root with:

```sh
python3 scripts/compile_latex.py /absolute/path/to/sep_evaluation_research_reference.tex \
  --compiler texlive --output-directory /absolute/path/to/docs/build
```

Run the command twice to resolve internal references. The PDF embeds the three disclosed historical-diagnostic figures from `audit_20260915/figures/`.
