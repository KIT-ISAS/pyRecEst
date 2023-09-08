import warnings

import numpy as np

from beartype import beartype
import typing


@beartype
def determine_all_deviations(
    results,
    extract_mean,
    distance_function: typing.Callable,
    groundtruths: np.ndarray[np.ndarray],
    mean_calculation_symm: str = "",
) -> np.ndarray:
    if mean_calculation_symm != "":
        raise NotImplementedError("Not implemented yet")

    assert (
        np.ndim(groundtruths) == 2
    ), "Assuming groundtruths to be a 3D array of shape (n_runs, n_timesteps, state_dimension)"

    all_deviations_last_mat = np.empty((len(results), groundtruths.shape[0]))

    for config, result_curr_config in enumerate(results):
        for run in range(len(groundtruths)):
            if "last_filter_states" not in result_curr_config:
                final_estimate = result_curr_config["last_estimates"][run]
            elif callable(extract_mean):
                final_estimate = extract_mean(
                    result_curr_config["last_filter_states"][run]
                )
            else:
                raise ValueError("No compatible mean extraction function given.")

            if final_estimate is not None:
                all_deviations_last_mat[config][run] = distance_function(
                    final_estimate, groundtruths[run, -1, :]
                )
            else:
                warnings.warn("No estimate for this filter, setting error to inf.")
                all_deviations_last_mat[config][run] = np.inf

        if np.any(np.isinf(all_deviations_last_mat[config])):
            print(
                f"Warning: {result_curr_config['filterName']} with {result_curr_config['filterParams']} "
                f"parameters apparently failed {np.sum(np.isinf(all_deviations_last_mat[config]))} "
                "times. Check if this is plausible."
            )

    return all_deviations_last_mat
