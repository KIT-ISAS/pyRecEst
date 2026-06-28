import unittest

from pyrecest.backend import allclose, array
from pyrecest.models import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel


class SimpleDist:
    def __init__(self, mean, samples, value):
        self._mean = mean
        self._samples = samples
        self.value = value
        self.last_arg = None

    def mean(self):
        return self._mean

    def sample(self, n):
        return self._samples[:n]

    def pdf(self, x):
        self.last_arg = x
        return self.value


def assert_close(test_case, actual, expected):
    test_case.assertTrue(bool(allclose(actual, expected)))


class AdditiveNoiseHelperKwargsTest(unittest.TestCase):
    def test_transition_helpers_forward_per_call_arguments(self):
        dist = SimpleDist(array([0.25]), array([[0.5]]), array(13.0))
        model = AdditiveNoiseTransitionModel(
            lambda x, dt, scale: array([x[0] + dt * scale]),
            noise_distribution=dist,
            dt=1.0,
            function_args={"scale": 2.0},
        )
        state = array([1.0])

        assert_close(self, model.mean(state, dt=0.5, scale=4.0), array([3.25]))
        assert_close(
            self,
            model.sample_next(state, n=1, dt=0.5, scale=4.0),
            array([[3.5]]),
        )
        assert_close(
            self,
            model.transition_density(array([3.75]), state, dt=0.5, scale=4.0),
            array(13.0),
        )
        assert_close(self, dist.last_arg, array([0.75]))

    def test_measurement_helpers_forward_per_call_arguments(self):
        dist = SimpleDist(array([0.25]), array([[0.5]]), array(17.0))
        model = AdditiveNoiseMeasurementModel(
            lambda x, scale, offset: array([scale * x[0] + offset]),
            noise_distribution=dist,
            function_args={"scale": 2.0, "offset": 0.0},
        )
        state = array([3.0])

        assert_close(
            self,
            model.predict_measurement(state, scale=4.0, offset=1.0),
            array([13.25]),
        )
        assert_close(self, model.mean(state, scale=4.0, offset=1.0), array([13.25]))
        assert_close(
            self,
            model.measurement_residual(array([14.5]), state, scale=4.0, offset=1.0),
            array([1.5]),
        )
        assert_close(
            self,
            model.sample_measurement(state, n=1, scale=4.0, offset=1.0),
            array([[13.5]]),
        )
        assert_close(
            self,
            model.likelihood(array([14.5]), state, scale=4.0, offset=1.0),
            array(17.0),
        )
        assert_close(self, dist.last_arg, array([1.5]))


if __name__ == "__main__":
    unittest.main()
