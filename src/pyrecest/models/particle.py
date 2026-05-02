class SampleableTransitionModel:
    def __init__(self, sample_next, function_is_vectorized=True):
        self.sample_next = sample_next
        self.function_is_vectorized = function_is_vectorized


class LikelihoodMeasurementModel:
    def __init__(self, likelihood):
        self.likelihood = likelihood
