# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import linalg, mean, vmap


def anees(estimates, uncertainties, groundtruths):
    n, dim = estimates.shape

    # Ensure the shapes of the inputs are correct
    assert uncertainties.shape == (n, dim, dim)
    assert groundtruths.shape == (n, dim)

    # Define a function to compute NEES for a single estimate
    def single_nees(estimate, uncertainty, groundtruth):
        error = estimate - groundtruth
        return error.T @ linalg.solve(uncertainty, error)

    # Vectorize the single_nees function over the batch dimension
    batch_nees = vmap(single_nees)

    NEES = batch_nees(estimates, uncertainties, groundtruths)
    return mean(NEES)
