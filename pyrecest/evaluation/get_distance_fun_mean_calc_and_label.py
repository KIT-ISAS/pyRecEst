import numpy as np
from numpy.linalg import norm
from pyrecest.distributions import AbstractHypertoroidalDistribution
from scipy.optimize import fminbound


# pylint: disable=too-many-branches
def get_distance_fun_mean_calc_and_label(mode, additional_params=None):
    if "circleSymm" in mode:
        error_label = "Error in radian"
        nSymm = int(
            mode.replace("circleSymm", "")
            .split("All")[0]
            .split("Highest")[0]
            .split("MinExpDev")[0]
        )
        symmetryOffsets = np.linspace(0, 2 * np.pi, nSymm + 1)
        symmetryOffsets = symmetryOffsets[:-1]

        if "Highest" in mode:

            def extract_mean(filterState):
                return fminbound(lambda x: -filterState["pdf"](x), 0, 2 * np.pi)

            def distance_function(xest, xtrue):
                return np.min(
                    AbstractHypertoroidalDistribution.angular_error(
                        xest, xtrue + symmetryOffsets.reshape(1, 1, -1)
                    ),
                    axis=2,
                )

        elif "All" in mode:
            raise NotImplementedError("Not implemented yet")

        elif "MinExpDev" in mode:

            def distance_function(xest, xtrue):
                return np.min(
                    AbstractHypertoroidalDistribution.angular_error(
                        xest, xtrue + symmetryOffsets.reshape(1, 1, -1)
                    ),
                    axis=2,
                )

            def extract_mean(filter_state):
                return fminbound(
                    lambda possibleEst: np.trapz(
                        distance_function(possibleEst, np.linspace(0, 2 * np.pi, 100))
                        * filter_state["pdf"](np.linspace(0, 2 * np.pi, 100)),
                        np.linspace(0, 2 * np.pi, 100),
                    ),
                    0,
                    2 * np.pi,
                )

    elif "circle" in mode or "hypertorus" in mode:

        def distance_function(xest, xtrue):
            return norm(AbstractHypertoroidalDistribution.angular_error(xest, xtrue))

        def extract_mean(filter_state):
            return filter_state.mean_direction()

        error_label = "Error in radian"

    elif "hypersphere" in mode:

        def distance_function(x1, x2):
            return np.arccos(np.dot(x1, x2))

        def extract_mean(filter_state):
            return filter_state.mean_direction()  # Stub

        error_label = "Error (orthodromic distance) in radian"

    elif "hypersphereSymmetric" in mode:

        def distance_function(x1, x2):
            return min(np.arccos(np.dot(x1, x2)), np.arccos(np.dot(x1, -x2)))

        extract_mean = "custom"
        error_label = "Angular error in radian"

    elif "se2" in mode or "se2linear" in mode:

        def distance_function(x1, x2):
            return norm(x1[1:3, :] - x2[1:3, :])

        def extract_mean(filter_state):
            raise NotImplementedError("Not implemented yet")

        error_label = "Error in meters"

    elif "se2bounded" in mode:

        def distance_function(xest, xtrue):
            return norm(
                AbstractHypertoroidalDistribution.angular_error(xest[0, :], xtrue[0, :])
            )  # Stub for angularError

        def extract_mean(filterState):
            raise NotImplementedError("Not implemented yet")

        error_label = "Error in radian"

    elif "se3" in mode or "se3linear" in mode:

        def distance_function(x1, x2):
            return norm(x1[4:7, :] - x2[4:7, :])

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

        error_label = "Error in meters"

    elif "se3bounded" in mode:

        def distance_function(x1, x2):
            return min(
                np.arccos(np.dot(x1[:4], x2[:4])), np.arccos(np.dot(x1[:4], -x2[:4]))
            )

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

        error_label = "Error in radian"

    elif "euclidean" in mode and "MTT" not in mode:

        def distance_function(x1, x2):
            return norm(x1 - x2)

        def extract_mean(filter_state):
            return filter_state.mean()

        error_label = "Error in meters"

    elif "MTTEuclidean" in mode:
        additional_params.get("cutoff_distance", 1000000)

        def distance_function(x1, x2):
            # Need to implement the OSPA distance function
            raise NotImplementedError("Not implemented yet")

        def extract_mean(_):
            raise NotImplementedError("Not implemented yet")

        error_label = "OSPA error in meters"

    else:
        raise ValueError("Mode not recognized")

    return distance_function, extract_mean, error_label
