# Research Scope

## Reproduction baseline

The active baseline is the COPT-MT method from *Can Computational Reducibility
Lead to Transferable Models for Graph Combinatorial Optimization?* No method
from NExCO, ASAP, or other later papers is implemented in the active pipeline.

The reproduction track consists of:

1. Existing reproductions of Tables 3, 5, and 7.
2. A clean reproduction of Table 15: an MIS model pretrained on RB-small is
   evaluated zero-shot on the complement of each DIMACS Maximum Clique graph.
3. The COPT sequential multi-seed decoder is used without patch inference or
   local search.

## Extension question

After Table 15 is reproduced, extensions must answer a controlled question:

> Does a representation learned on small graphs provide a more useful vertex
> ranking than inexpensive structural heuristics, and can that ranking improve
> anytime or exact Maximum Clique search on larger unseen graphs?

Reproduction and extensions must never share result tables without an explicit
method label. Experimental extensions belong under `experiments/extensions/`
and must not change default COPT task configuration.

## Claim boundary

- Feasibility validation certifies that a returned set is a clique; it does not
  certify maximum cardinality.
- A neural ordering may guide an exact solver without compromising exactness as
  long as it changes search order only and is not used for unsafe pruning.
- DIMACS is a benchmark suite, not sufficient evidence of performance on
  application-derived real-world graphs.

