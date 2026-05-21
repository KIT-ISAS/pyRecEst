# Performance And Benchmarking

Benchmarks should report both numerical outputs and timing information. The
script `benchmarks/basic_regressions.py` emits JSON that can be archived by CI and
compared across releases.

Backend-specific targets should be explicit:

| Backend | Performance target                                                        |
|---------|---------------------------------------------------------------------------|
| NumPy   | Reliable default behavior and SciPy-heavy workflows.                      |
| PyTorch | Tensor/autodiff workflows and GPU-capable native paths where implemented. |
| JAX     | Pure functional and vectorized workflows where JIT is practical.          |

Avoid optimizing a backend-specific path until its dtype, device, and autodiff
semantics are documented in the capability matrix.

## Historical Benchmarks

Use `asv` for trend tracking across commits and releases:

```bash
poetry run asv run --quick
poetry run asv publish
poetry run asv preview
```

The lightweight JSON benchmark remains useful for CI smoke checks, while ASV is
better for longitudinal analysis of algorithmic changes. Add benchmarks for new
performance-sensitive APIs before optimizing them.
