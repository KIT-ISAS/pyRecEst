import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import atleast_2d, empty_like, squeeze


# pylint: disable=too-many-branches
def generate_groundtruth(simulation_param, x0=None):
    """
    Generate ground truth based on the given scenario parameters.

    Parameters:
        simulation_param (dict): Dictionary containing scenario parameters.
        x0 (ndarray): Starting point (optional).

    Returns:
        groundtruth (np.ndarray[np.ndarray]): Generated ground truth as an
        array of arrays (!) because the size of the ground truth is not
        necessarily the same over time (e.g., if the number of targets changes)
    """
    if x0 is None:
        x0 = simulation_param["initial_prior"].sample(simulation_param["n_targets"])

    assert (
        x0.ndim == 1
        and simulation_param["n_targets"] == 1
        or x0.shape[0] == simulation_param["n_targets"]
    ), "Mismatch in number of targets."

    # Initialize ground truth
    groundtruth = np.empty(simulation_param["n_timesteps"], dtype=object)

    if "inputs" in simulation_param:
        assert (
            simulation_param["inputs"].shape[1] == simulation_param["n_timesteps"] - 1
        ), "Mismatch in number of timesteps."

    groundtruth[0] = atleast_2d(x0)

    for t in range(1, simulation_param["n_timesteps"]):
        groundtruth[t] = empty_like(groundtruth[0])
        for target_no in range(simulation_param["n_targets"]):
            if "gen_next_state_with_noise" in simulation_param:
                if (
                    "inputs" not in simulation_param
                    or simulation_param["inputs"] is None
                ):
                    groundtruth[t][target_no, :] = simulation_param[
                        "gen_next_state_with_noise"
                    ](groundtruth[t - 1, target_no, :])
                else:
                    groundtruth[t][target_no, :] = simulation_param[
                        "gen_next_state_with_noise"
                    ](
                        groundtruth[t - 1][target_no, :],
                        simulation_param["inputs"][t - 1, :],
                    )

            elif "sys_noise" in simulation_param:
                if "gen_next_state_without_noise" in simulation_param:
                    if (
                        "inputs" not in simulation_param
                        or simulation_param["inputs"] is None
                    ):
                        state_to_add_noise_to = simulation_param[
                            "gen_next_state_without_noise"
                        ](groundtruth[t - 1, target_no, :])
                    else:
                        state_to_add_noise_to = simulation_param[
                            "gen_next_state_without_noise"
                        ](
                            groundtruth[:, t - 1, target_no],
                            simulation_param["inputs"][t - 1, :],
                        )
                else:
                    assert (
                        "inputs" not in simulation_param
                        or simulation_param["inputs"] is None
                    ), "No inputs accepted for the identity system model."
                    state_to_add_noise_to = groundtruth[t - 1][target_no, :]

                groundtruth[t][target_no, :] = state_to_add_noise_to + simulation_param[
                    "sys_noise"
                ].sample(1)

            else:
                raise ValueError("Cannot generate groundtruth.")

    assert groundtruth[0].shape[0] == simulation_param["n_targets"]
    assert groundtruth[0].shape[1] == simulation_param["initial_prior"].dim
    for t in range(simulation_param["n_timesteps"]):
        groundtruth[t] = squeeze(groundtruth[t])

    return groundtruth
