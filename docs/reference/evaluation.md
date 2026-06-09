# Evaluation

## Model-comparison evidence margins

`pyrecest.evaluation.model_comparison` contains lightweight, dataframe-oriented
helpers for comparing model-evidence tables. The helpers are intentionally
domain-neutral: callers provide the model labels, grouping columns, thresholds,
and cluster/group identifiers.

Useful entry points include:

- `paired_model_margin_decisions`, which compares a positive model against a
  reference model and separates positive claims, reference claims, and ambiguous
  events using a symmetric log-evidence threshold;
- `paired_model_margin_summary`, which summarizes raw wins, confident claims,
  ambiguous events, and mean/median log-evidence margins;
- `paired_model_margin_threshold_sweep`, which evaluates the paired decision
  rule over multiple candidate thresholds;
- `select_paired_model_margin_threshold`, which selects the smallest threshold
  satisfying synthetic false-positive and recall constraints;
- `leave_one_group_out_summary`, a generic leave-one-group-out wrapper for
  grouped robustness checks;
- `cluster_bootstrap_margin_summary`, which returns cluster-resampled
  uncertainty intervals for raw-win fractions, claim fractions, and evidence
  margins; and
- `grouped_claim_gate_summary`, which summarizes whether every group satisfies
  majority/no-forbidden-claim/positive-margin gates.

These utilities are useful for model comparison, parameter selection, and
paper-quality diagnostics whenever multiple filters, smoothers, or trackers emit
comparable log marginal likelihoods.

## Pareto and equal-quality selection

`pyrecest.evaluation.pareto` contains small dataframe-oriented utilities for
rate--distortion and equal-quality comparisons.  Callers provide objective
columns and objective directions, so the helpers are intentionally domain-neutral
and can be used for particle-count, runtime, storage-size, or accuracy/quality
trade-offs.

Useful entry points include `pareto_front_indices`, `is_pareto_front`,
`record_dominates`, `constraint_mask`, `select_under_constraints`, and
`equal_quality_selection`.

## Implicit-surface helpers

`pyrecest.evaluation.implicit_surfaces` contains lightweight helpers for
backend-neutral scalar-field and implicit-surface evaluation. These helpers cover
residual extraction through structural `value(points)` objects, surface-band
masks, inside/outside classification, and surface-band probabilities from signed
distance means and standard deviations. They are useful for shape-estimation and
extended-object diagnostics without requiring implementations to inherit from a
PyRecEst base class.

## Protected-tail selection helpers

`pyrecest.evaluation.selection` contains deterministic, domain-neutral helpers
for selecting a fixed-size subset under reliability or confidence constraints.
The helpers are useful when an evaluation or ablation should preserve a bounded
number of low-reliability hypotheses, measurements, particles, or shape samples
while still ranking each region by a primary score. They intentionally avoid
domain-specific names such as visibility, splats, or rendering; callers provide
the primary scores, tail scores, reliability scores, retention fractions, and
tail quantiles.

::: pyrecest.evaluation
