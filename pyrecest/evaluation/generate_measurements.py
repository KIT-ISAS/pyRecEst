import numpy as np
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    GaussianDistribution,
    VonMisesFisherDistribution,
    WatsonDistribution,
)


# pylint: disable=too-many-branches
def generate_measurements(groundtruth, simulation_param):
    """
    Generate measurements based on the given groundtruth and scenario parameters.

    Parameters:
        groundtruth (ndarray): Ground truth data.
        simulation_param (dict): Dictionary containing scenario parameters.

    Returns:
        measurements (list): List of generated measurements at each time step.
        Is a list because the number of measurements can vary at each time step.
        Comprises timesteps elements, each of which is a numpy array of shape
        (n_meas_at_individual_time_step[t], n_dim).
    """
    assert (
        np.shape(simulation_param["n_meas_at_individual_time_step"])
        == (simulation_param["n_timesteps"],)
    )
    measurements = np.empty(simulation_param["n_timesteps"], dtype=object)
    
    if simulation_param.get("MTT", False) and simulation_param.get("EOT", False):
        raise NotImplementedError("Multiple extended object tracking is currently not supported.")
    if simulation_param.get("EOT", False):
        assert "target_shape" in simulation_param
        raise NotImplementedError("Extended object tracking is currently not supported.")
    
    if simulation_param.get("MTT", False):
        assert simulation_param["clutter_rate"] == 0, "Clutter currently not supported."

        n_observations = np.random.binomial(
            1,
            simulation_param["detection_probability"],
            (simulation_param["n_timesteps"], simulation_param["n_targets"]),
        )

        for t in range(simulation_param["n_timesteps"]):
            n_meas_at_t = np.sum(n_observations[t, :])
            measurements[t] = np.nan * np.zeros(
                (simulation_param["meas_matrix_for_each_target"].shape[0], n_meas_at_t)
            )

            meas_no = 0
            for target_no in range(simulation_param["n_targets"]):
                if n_observations[t, target_no] == 1:
                    meas_no += 1
                    measurements[t][meas_no - 1, :] = np.dot(
                        simulation_param["meas_matrix_for_each_target"],
                        groundtruth[t, target_no, :],
                    ) + simulation_param["meas_noise"].sample(1)
                else:
                    assert (
                        n_observations[t, target_no] == 0
                    ), "Multiple measurements currently not supported."

            assert meas_no == n_meas_at_t, "Mismatch in number of measurements."

    else:
        if "meas_generator" in simulation_param:
            raise NotImplementedError(
                "Scenarios based on a 'measGenerator' are currently not supported."
            )
        for t in range(simulation_param["n_timesteps"]):
            n_meas = simulation_param["n_meas_at_individual_time_step"][t]
            meas_noise = simulation_param["meas_noise"]

            if isinstance(meas_noise, AbstractHypertoroidalDistribution):
                noise_samples = meas_noise.sample(n_meas)
                measurements[t] = np.mod(
                    np.squeeze(
                        np.tile(
                            groundtruth[t - 1],
                            (
                                n_meas,
                                1,
                            ),
                        )
                        + noise_samples
                    ),
                    2 * np.pi,
                )

            elif isinstance(
                meas_noise, (VonMisesFisherDistribution, WatsonDistribution)
            ):
                curr_dist = meas_noise
                curr_dist.mu = groundtruth[t - 1]
                measurements[t] = curr_dist.sample(n_meas)

            elif isinstance(meas_noise, GaussianDistribution):
                noise_samples = meas_noise.sample(n_meas)
                measurements[t] = np.squeeze(
                    np.tile(
                        groundtruth[t - 1],
                        (
                            n_meas,
                            1,
                        ),
                    )
                    + noise_samples
                )

    return measurements
