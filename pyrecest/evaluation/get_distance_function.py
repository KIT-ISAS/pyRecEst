import numpy as np
from numpy.linalg import norm
from pyrecest.distributions import AbstractHypertoroidalDistribution


def get_distance_function(mode, additional_params=None, nSymm=None, symmetryOffsets=None):
    if nSymm is not None or symmetryOffsets is not None or "Symm" in mode:
        raise NotImplementedError("Not implemented yet")

    if "circle" in mode or "hypertorus" in mode:
        def distance_function(xest, xtrue):
            return norm(AbstractHypertoroidalDistribution.angular_error(xest, xtrue))

    elif "hypersphere" in mode:
        def distance_function(x1, x2):
            return np.arccos(np.dot(x1, x2))

    elif "hypersphereSymmetric" in mode:
        def distance_function(x1, x2):
            return min(np.arccos(np.dot(x1, x2)), np.arccos(np.dot(x1, -x2)))

    elif "se2" in mode or "se2linear" in mode:
        def distance_function(x1, x2):
            return norm(x1[1:3, :] - x2[1:3, :])

    elif "se2bounded" in mode:
        def distance_function(xest, xtrue):
            return norm(AbstractHypertoroidalDistribution.angular_error(xest[0, :], xtrue[0, :]))

    elif "se3" in mode or "se3linear" in mode:
        def distance_function(x1, x2):
            return norm(x1[4:7, :] - x2[4:7, :])

    elif "se3bounded" in mode:
        def distance_function(x1, x2):
            return min(
                np.arccos(np.dot(x1[:4], x2[:4])),
                np.arccos(np.dot(x1[:4], -x2[:4]))
            )

    elif "euclidean" in mode and "MTT" not in mode:
        def distance_function(x1, x2):
            return norm(x1 - x2)

    elif "MTTEuclidean" in mode:
        additional_params.get("cutoff_distance", 1000000)
        def distance_function(x1, x2):
            raise NotImplementedError("Not implemented yet")

    else:
        raise ValueError("Mode not recognized")
    
    return distance_function
