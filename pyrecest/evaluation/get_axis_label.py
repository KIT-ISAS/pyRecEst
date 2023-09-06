def get_axis_label(mode):
    if "circleSymm" in mode:
        error_label = "Error in radian"

    elif "circle" in mode or "hypertorus" in mode:
        error_label = "Error in radian"

    elif "hypersphere" in mode:
        error_label = "Error (orthodromic distance) in radian"

    elif "hypersphereSymmetric" in mode:
        error_label = "Angular error in radian"

    elif "se2" in mode or "se2linear" in mode:
        error_label = "Error in meters"

    elif "se2bounded" in mode:
        error_label = "Error in radian"

    elif "se3" in mode or "se3linear" in mode:
        error_label = "Error in meters"

    elif "se3bounded" in mode:
        error_label = "Error in radian"

    elif "euclidean" in mode and "MTT" not in mode:
        error_label = "Error in meters"

    elif "MTTEuclidean" in mode:
        error_label = "OSPA error in meters"

    else:
        raise ValueError("Mode not recognized")

    return error_label