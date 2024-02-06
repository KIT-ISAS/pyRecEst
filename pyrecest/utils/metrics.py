# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import linalg, mean, vmap
from shapely import Polygon


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


def iou_polygon(polygon1, polygon2):
    if not isinstance(polygon1, Polygon):
        polygon1 = Polygon(polygon1)
    if not isinstance(polygon2, Polygon):
        polygon2 = Polygon(polygon2)

    # Compute the intersection and union
    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)
    
    iou = intersection.area / union.area if union.area > 0 else 0
    
    return iou