# Sparse second-order grid evidence

`pyrecest.filters.sparse_second_order_grid_evidence` computes exact evidence for
finite-support second-order grid hidden Markov models.  The latent state is a
sparse pair `(x[t-1], x[t])`; callers provide the initial pair lattice and sparse
transition rows.

The primitive is useful when a dense `n_states^2 x n_states` transition would be
too large, but the model has finite local support.  It returns the normalized
log evidence, terminal single-state posterior, optional fixed-interval smoothed
single-state marginals, and support/cache diagnostics.

The implementation is intentionally domain-neutral: replay, map matching,
maneuvering target tracking, and other second-order grid models can supply their
own transition-row builder while sharing the same forward/backward evidence
calculation.

## Evidence-only mode

Set `evidence_mode="evidence_only"` or `return_smoothed=False` when a model
comparison or parameter sweep only needs `log p(y_1:T | model)`.  The forward
recursion and terminal posterior are still computed, but the backward smoother
and smoothed time-marginals are skipped.

The log evidence must match full-smoothing mode to numerical precision.  The
diagnostics include `evidence_computation_mode`, `evidence_only`,
`smoothed_posterior_returned`, and `terminal_posterior_returned` so downstream
artifact tables can make the output mode explicit.
