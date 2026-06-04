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

::: pyrecest.evaluation
