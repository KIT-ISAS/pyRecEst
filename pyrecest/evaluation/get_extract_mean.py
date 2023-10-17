def get_extract_mean(manifold_name, mtt_scenario=False):
    if "symm" in manifold_name:
        raise NotImplementedError("Not implemented yet")

    if "circle" in manifold_name or "hypertorus" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "hypersphere" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()  # Stub

    elif "hypersphereSymmetric" in manifold_name:
        extract_mean = "custom"

    elif "se2" in manifold_name or "se2linear" in manifold_name:
        raise NotImplementedError("Not implemented yet")

    elif "se2bounded" in manifold_name:
        raise NotImplementedError("Not implemented yet")

    elif "se3" in manifold_name or "se3linear" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "se3bounded" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and not mtt_scenario:

        def extract_mean(filter_state):
            return filter_state.mean()

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and mtt_scenario:

        def extract_mean(_):
            raise NotImplementedError("Not implemented yet")

    else:
        raise ValueError("Mode not recognized")

    return extract_mean