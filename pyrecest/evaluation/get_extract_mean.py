def get_extract_mean(mode):
    if "symm" in mode:
        raise NotImplementedError("Not implemented yet")

    if "circle" in mode or "hypertorus" in mode:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "hypersphere" in mode:

        def extract_mean(filter_state):
            return filter_state.mean_direction()  # Stub

    elif "hypersphereSymmetric" in mode:
        extract_mean = "custom"

    elif "se2" in mode or "se2linear" in mode:
        raise NotImplementedError("Not implemented yet")

    elif "se2bounded" in mode:
        raise NotImplementedError("Not implemented yet")

    elif "se3" in mode or "se3linear" in mode:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "se3bounded" in mode:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "euclidean" in mode and "MTT" not in mode:

        def extract_mean(filter_state):
            return filter_state.mean()

    elif "MTTEuclidean" in mode:

        def extract_mean(_):
            raise NotImplementedError("Not implemented yet")

    else:
        raise ValueError("Mode not recognized")

    return extract_mean
