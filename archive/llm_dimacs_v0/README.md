# Archived DIMACS Patch Prototype (v0)

This directory preserves the first exploratory DIMACS extension generated with
LLM assistance. It is retained for traceability, not as an active experiment.

## Scientific status

- The implementation uses the COPT-MT backbone and custom engineering code.
- It does **not** faithfully implement NExCO, ASAP, or the framework proposed in
  *Certified Correctness in Neural Constraint Reasoning Requires Symbolic
  Integration*. Those papers were discussed only in the archived strategy text.
- The patch inference implementation has a local-to-global ordering defect after
  top-degree truncation and severe patch overlap on dense DIMACS graphs.
- The old Mode C result retained Mode B whenever fine-tuning was worse, so the
  reported statement `Mode C >= Mode B` was true by construction.
- The old benchmark used an MIS head on the original graph. The COPT paper's
  Table 15 instead evaluates MIS on the complement graph.

Do not use the archived report as evidence for a research claim. Raw outputs are
kept only to make the development history reproducible.

