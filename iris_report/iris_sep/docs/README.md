# Research-reference papers

## IB Physics-level scaffold

- Source: [`iris_sep_ib_physics_research_paper.tex`](iris_sep_ib_physics_research_paper.tex)
- Purpose: a simpler, evidence-locked research-paper scaffold centred on the episode-normalized/onset finding, with each major finding tied to why it is useful, plus negative results, novelty boundaries, limitations, and the next experiment.
- Level: written so that the core argument can be defended with IB Physics plus basic statistics (TSS, weighted averages, and bootstrap intuition), without relying on the advanced covariance/bounds derivations in the full audit reference.

## Full technical audit reference

- Source: [`sep_evaluation_research_reference.tex`](sep_evaluation_research_reference.tex)
- Compiled PDF: [`build/sep_evaluation_research_reference.pdf`](build/sep_evaluation_research_reference.pdf)

Both are AI-authored internal references built from the audited evidence package. They are not student-authored initial submission prose. Student authors must verify the evidence, understand the calculations, rewrite competition materials in their own words, and disclose assistance as required by the applicable fair rules.

Compile from the LaTeX plugin root with:

```sh
python3 scripts/compile_latex.py /absolute/path/to/iris_sep_ib_physics_research_paper.tex \
  --compiler texlive --output-directory /absolute/path/to/docs/build
```

Run the command twice to resolve internal references. The scaffold reuses the three disclosed historical-diagnostic figures from `audit_20260915/figures/`.
