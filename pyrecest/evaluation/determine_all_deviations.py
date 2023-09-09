import warnings
from typing import Callable

import numpy as np
from beartype import beartype


@beartype
def determine_all_deviations(
    results,
    extract_mean,
    distance_function: Callable,
    groundtruths: np.ndarray,
    mean_calculation_symm: str = "",
) -> np.ndarray:
    if mean_calculation_symm != "":
        raise NotImplementedError("Not implemented yet")

    assert (
        np.ndim(groundtruths) == 2
        and isinstance(groundtruths[0, 0], np.ndarray)
        and np.ndim(groundtruths[0, 0])
        in (
            1,
            2,
        )
    ), "Assuming groundtruths to be a 2-D array of shape (n_runs, n_timesteps) composed arrays of shape (n_dim,) or (n_targets,n_dim)."

    all_deviations_last_mat = np.empty((len(results), groundtruths.shape[0]))

    for config_no, result_curr_config in enumerate(results):
        for run in range(len(groundtruths)):
            if isinstance(result_curr_config[run], np.ndarray):
                # If estimates are already given as numpy array, use it
                final_estimate = result_curr_config[run]
            elif callable(extract_mean):
                # Otherwise, use extract_mean to obtain the estimate
                final_estimate = extract_mean(result_curr_config[run])
            else:
                raise ValueError("No compatible mean extraction function given.")

            if final_estimate is not None:
                all_deviations_last_mat[config_no][run] = distance_function(
                    final_estimate, groundtruths[run, -1]
                )
            else:
                warnings.warn("No estimate for this filter, setting error to inf.")
                all_deviations_last_mat[config_no][run] = np.inf

        if np.any(np.isinf(all_deviations_last_mat[config_no])):
            print(
                f"Warning: {result_curr_config['filterName']} with {result_curr_config['filterParams']} "
                f"parameters apparently failed {np.sum(np.isinf(all_deviations_last_mat[config_no]))} "
                "times. Check if this is plausible."
            )

    return all_deviations_last_mat
