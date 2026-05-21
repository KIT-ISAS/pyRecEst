# Performance And Benchmarking

Benchmarks should report both numerical outputs and timing information. The
script `benchmarks/basic_regressions.py` emits JSON that can be archived by CI and
compared across releases. The companion script
`scripts/check_benchmark_results.py` compares deterministic benchmark outputs
against JSON baselines under `benchmarks/baselines/`.

Baseline checks should start by enforcing numerical outputs only. Runtime
thresholds can be added later with `max_elapsed_seconds`, `elapsed_seconds` plus
`--max-runtime-ratio`, or `--warn-only-runtime` when the goal is to collect early
signals without failing CI on shared-runner noise.

Backend-specific targets should be explicit:

| Backend | Performance target                                                        |
|---------|---------------------------------------------------------------------------|
| NumPy   | Reliable default behavior and SciPy-heavy workflows.                      |
| PyTorch | Tensor/autodiff workflows and GPU-capable native paths where implemented. |
| JAX     | Pure functional and vectorized workflows where JIT is practical.          |

Avoid optimizing a backend-specific path until its dtype, device, and autodiff
semantics are documented in the capability matrix.
