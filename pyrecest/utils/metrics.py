from pyrecest.backend import linalg, mean, zeros


def anees(estimates, uncertainties, groundtruths):
    n, dim = estimates.shape

    # Ensure the shapes of the inputs are correct
    assert uncertainties.shape == (n, dim, dim)
    assert groundtruths.shape == (n, dim)

    NEES = zeros(n)

    for i in range(n):
        error = estimates[i] - groundtruths[i]
        NEES[i] = error.T @ linalg.solve(uncertainties[i], error)

    return mean(NEES)
