# Sparse transition-row cache

`pyrecest.filters.SparseTransitionRowCache` is a small utility for finite-support
grid filters whose transition rows are expensive to build but frequently reused.

It is intentionally domain-neutral. A row is represented as:

```python
(destination_indices, weights)
```

and callers define the cache key.

```python
from pyrecest.filters import SparseTransitionRowCache

cache = SparseTransitionRowCache()

dst, weights = cache.get_or_build(
    key=("mode", 1, "source", 42),
    builder=lambda: expensive_transition_row(),
)

print(cache.diagnostics())
```

The cache does not normalize or validate rows. That responsibility belongs to
the filter or model-evidence routine consuming the rows. This keeps the cache
useful for Euclidean grids, manifold grids, pair-state HMMs, and custom finite
support transitions.

The helper `cached_sparse_transition_rows` applies the same idea to a batch of
source states:

```python
rows, cache = cached_sparse_transition_rows(
    source_states,
    row_builder=lambda state: build_row(state),
    cache_key_builder=lambda state: tuple(state),
)
```

The sparse second-order grid evidence primitive accepts an external cache via
`transition_row_cache`, which lets callers keep row-reuse diagnostics or share
a cache across repeated evidence calls with the same transition semantics.
