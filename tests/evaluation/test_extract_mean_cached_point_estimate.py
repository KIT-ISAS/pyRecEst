from pyrecest.evaluation.get_extract_mean import get_extract_mean


class _CachedPointEstimateState:
    def __init__(self, value):
        self.get_point_estimate = value


def test_euclidean_extract_mean_accepts_cached_point_estimate_attribute():
    extractor = get_extract_mean("euclidean")

    assert extractor(_CachedPointEstimateState("cached-estimate")) == "cached-estimate"
