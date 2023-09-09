import warnings
from typing import Any, Dict

import numpy as np

from .generate_groundtruth import generate_groundtruth
from .generate_measurements import generate_measurements
from .perform_predict_update_cycles import perform_predict_update_cycles


# pylint: disable=R0913,R0914,W0718,R0912
def iterate_configs_and_runs(
    scenario_param: Dict[str, Any],
    filter_configs: list[Dict[str, None | int | list[Any]]],
    n_runs: int,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    tolerate_failure: bool = False,
    auto_warning_on_off: bool = True,
):
    if extract_all_point_estimates:
        warnings.warn(
            "Extracting all point estimates can have a massive impact on the run time. Use this for debugging only"
        )
        raise NotImplementedError("This is not implemented yet.")

    n_configs = sum(np.size(f["parameter"]) for f in filter_configs)
    run_times = np.empty((n_configs, n_runs))

    # Creating an array of arrays allows for different numbers of targets and measurements per step or run
    groundtruths = np.empty((n_runs, scenario_param["timesteps"]), dtype=np.ndarray)
    measurements = np.empty((n_runs, scenario_param["timesteps"]), dtype=np.ndarray)

    run_failed = np.zeros((n_configs, n_runs), dtype=bool)

    if convert_to_point_estimate_during_runtime:
        raise NotImplementedError("This is not implemented yet.")

    last_filter_states = np.empty((n_configs, n_runs), dtype=object)

    for run in range(n_runs):
        groundtruths[run, :] = generate_groundtruth(scenario_param)
        measurements[run, :] = generate_measurements(
            groundtruths[run, :], scenario_param
        )

        for config_no, filter_config in enumerate(filter_configs):
            try:
                if (
                    not convert_to_point_estimate_during_runtime
                    and not extract_all_point_estimates
                ):
                    (
                        last_filter_states[config_no, run],
                        run_times[config_no, run],
                        *_,
                    ) = perform_predict_update_cycles(
                        scenario_param=scenario_param,
                        filter_config=filter_config,
                        groundtruth=groundtruths[run, :],
                        measurements=measurements[run, :],
                    )

                elif (
                    not convert_to_point_estimate_during_runtime
                    and extract_all_point_estimates
                ):
                    raise NotImplementedError("This is not implemented yet.")
                elif (
                    convert_to_point_estimate_during_runtime
                    and not extract_all_point_estimates
                ):
                    raise NotImplementedError("This is not implemented yet.")
                elif (
                    convert_to_point_estimate_during_runtime
                    and extract_all_point_estimates
                ):
                    raise NotImplementedError("This is not implemented yet.")
                else:
                    raise ValueError("This should not happen.")

            except Exception as err:
                if not tolerate_failure:
                    raise err
                run_failed[config_no, run] = True
                if auto_warning_on_off:
                    warnings.warn(
                        f"Filter {config_no} config {filter_config['parameter']} run {run} FAILED: {str(err)}"
                    )

    return last_filter_states, run_times, run_failed, groundtruths, measurements
