import warnings

import numpy as np

from .generate_groundtruth import generate_groundtruth
from .generate_measurements import generate_measurements
from .perform_predict_update_cycles import perform_predict_update_cycles


# pylint: disable=R0913,R0914,W0718,R0912
def iterate_configs_and_runs(
    scenario_param,
    filters,
    n_runs,
    convert_to_point_estimate_during_runtime=False,
    extract_all_point_estimates=False,
    tolerate_failure=False,
    auto_warning_on_off=True,
):
    if extract_all_point_estimates:
        warnings.warn(
            "Extracting all point estimates can have a massive impact on the run time. Use this for debugging only"
        )
        raise NotImplementedError("This is not implemented yet.")

    n_configs = sum(np.size(f["filter_params"]) for f in filters)
    t = np.empty((n_configs, n_runs))

    groundtruths = [None] * n_runs
    measurements = np.empty((n_runs, scenario_param["timesteps"]), dtype=object)

    run_failed = np.zeros((n_configs, n_runs), dtype=bool)

    if convert_to_point_estimate_during_runtime:
        np.empty((n_configs, n_runs), dtype=object)
    else:
        last_filter_states = np.empty((n_configs, n_runs), dtype=object)

    curr_config_index = 0

    for r in range(n_runs):
        groundtruths[r] = generate_groundtruth(scenario_param)
        measurements[r, :] = generate_measurements(groundtruths[r], scenario_param)

    for filter_no, filter_data in enumerate(filters):
        if filter_data["filter_params"] is None or isinstance(
            filter_data["filter_params"], int
        ):
            filter_data["filter_params"] = [
                filter_data["filter_params"]
            ]  # To make it iterable
        for config in filter_data["filter_params"]:
            filter_param = {"name": filter_data["name"], "parameter": config}

            try:
                if (
                    not convert_to_point_estimate_during_runtime
                    and not extract_all_point_estimates
                ):
                    last_filter_states[
                        curr_config_index, r
                    ] = perform_predict_update_cycles(
                        scenario_param=scenario_param,
                        filter_param=filter_param,
                        groundtruth=groundtruths[r],
                        measurements=measurements[r, :],
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

                if auto_warning_on_off:
                    warnings.warn(
                        f"Filter {filter_no} config {config} run {r} FAILED: {str(err)}"
                    )

                run_failed[curr_config_index, r] = True

            curr_config_index += 1

    return t, groundtruths, measurements
