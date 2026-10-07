# Reproduction Experiments

This directory documents paper-faithful experiments. Reproduction code may use
the original COPT implementation and evaluation-only utilities, but it must not
depend on experimental patch inference, test-time fine-tuning, or local search.

## Table 15

Run the full-graph MIS-on-complement protocol with an official RB-small MIS
checkpoint:

```powershell
python scripts/reproduce_table15.py `
  --checkpoint path/to/official/rb-small-mis.ckpt `
  --mode pretrained
```

The script refuses checkpoints without the required `mis` output and records
the Git commit, checkpoint SHA-256, runtime breakdown, and difference from the
published result for every instance.

