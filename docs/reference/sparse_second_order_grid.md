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
