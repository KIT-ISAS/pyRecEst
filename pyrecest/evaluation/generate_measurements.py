import numpy as np
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    GaussianDistribution,
    VonMisesFisherDistribution,
    WatsonDistribution,
)


def generate_measurements(groundtruth, scenario_param):
    """
    Generate measurements based on the given groundtruth and scenario parameters.

    Parameters:
        groundtruth (ndarray): Ground truth data.
        scenario_param (dict): Dictionary containing scenario parameters.

    Returns:
        measurements (list): List of generated measurements at each time step.
        Is a list because the number of measurements can vary at each time step.
        Comprises timesteps elements, each of which is a numpy array of shape
        (n_meas_at_individual_time_step[t], n_dim).
    """
    measurements = [None] * scenario_param["timesteps"]
    if "MTT" in scenario_param.get("manifold_type", ""):
        assert scenario_param["clutter_rate"] == 0, "Clutter currently not supported."

        n_observations = np.random.binomial(
            1,
            scenario_param["detection_probability"],
            (scenario_param["timesteps"], scenario_param["n_targets"]),
        )

        for t in range(scenario_param["timesteps"]):
            n_meas_at_t = np.sum(n_observations[t, :])
            measurements[t] = np.nan * np.zeros(
                (scenario_param["meas_matrix_for_each_target"].shape[0], n_meas_at_t)
            )

            meas_no = 0
            for target_no in range(scenario_param["n_targets"]):
                if n_observations[t, target_no] == 1:
                    meas_no += 1
                    measurements[t][meas_no - 1, :] = np.dot(
                        scenario_param["meas_matrix_for_each_target"],
                        groundtruth[t, target_no, :],
                    ) + scenario_param["meas_noise"].sample(1)
                else:
                    assert (
                        n_observations[t, target_no] == 0
                    ), "Multiple measurements currently not supported."

            assert meas_no == n_meas_at_t, "Mismatch in number of measurements."

    else:
        if "meas_generator" in scenario_param:
            raise NotImplementedError(
                "Scenarios based on a 'measGenerator' are currently not supported."
            )
        for t in range(scenario_param["timesteps"]):
            n_meas = scenario_param["n_meas_at_individual_time_step"][t]
            meas_noise = scenario_param["meas_noise"]

            if isinstance(meas_noise, AbstractHypertoroidalDistribution):
                noise_sample = meas_noise.sample(n_meas)
                measurements[t] = np.mod(
                    np.tile(groundtruth[t - 1, :], (1, n_meas)) + noise_sample,
                    2 * np.pi,
                )

            elif isinstance(meas_noise, (VonMisesFisherDistribution, WatsonDistribution)):
                curr_dist = meas_noise
                curr_dist.mu = groundtruth[t - 1]
                measurements[t] = curr_dist.sample(n_meas)

            elif isinstance(meas_noise, GaussianDistribution):
                noise_sample = meas_noise.sample(n_meas)
                measurements[t] = (
                    np.tile(groundtruth[t - 1], (n_meas, 1)) + noise_sample
                )

    return measurements
