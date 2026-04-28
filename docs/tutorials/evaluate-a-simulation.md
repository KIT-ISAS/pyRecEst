# Evaluate A Simulation

PyRecEst includes evaluation helpers for generating simulated scenarios, running
filters, saving the raw evaluation output, and summarizing errors and runtimes.

This tutorial runs the built-in `R2randomWalk` scenario with a Kalman filter.
The short run count keeps the example fast; larger experiments should use many
more runs.

```python
import tempfile
import warnings

from pyrecest.evaluation import (
    evaluate_for_simulation_config,
    summarize_filter_results,
)


filter_configs = [{"name": "kf", "parameter": None}]

with tempfile.TemporaryDirectory() as output_dir:
    (
        last_filter_states,
        runtimes,
        run_failed,
        groundtruths,
        measurements,
        scenario_config,
        expanded_filter_configs,
        evaluation_config,
    ) = evaluate_for_simulation_config(
        "R2randomWalk",
        filter_configs,
        n_runs=3,
        n_timesteps=5,
        initial_seed=1,
        save_folder=output_dir,
        auto_warning_on_off=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        summaries = summarize_filter_results(
            scenario_config=scenario_config,
            filter_configs=expanded_filter_configs,
            runtimes=runtimes,
            groundtruths=groundtruths,
            run_failed=run_failed,
            last_filter_states=last_filter_states,
        )

for result in summaries:
    print(
        f"{result['name']}: "
        f"error_mean={float(result['error_mean']):.3f}, "
        f"error_std={float(result['error_std']):.3f}, "
        f"time_mean={float(result['time_mean']):.4f}, "
        f"failure_rate={float(result['failure_rate']):.3f}"
    )
```

## What To Notice

- `evaluate_for_simulation_config()` accepts either a built-in scenario name or
  a scenario configuration dictionary.
- `filter_configs` use compact filter names such as `kf` for Kalman filter and
  `pf` for particle filter.
- A list or tuple in `parameter` expands into multiple filter configurations.
  For example, `{"name": "pf", "parameter": [51, 81]}` runs two particle-filter
  settings.
- The function saves a timestamped `.npy` file in `save_folder` and also returns
  the arrays needed for immediate analysis.
- `summarize_filter_results()` computes aggregate error, runtime, and failure
  statistics from the returned evaluation data.

## Useful Variations

- Increase `n_runs` for more reliable summary statistics.
- Set `extract_all_point_estimates=True` when you need estimates at every time
  step instead of only final states.
- Use `evaluate_for_file()` when ground truth and measurements already exist in
  a saved data file.
