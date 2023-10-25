import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution
from pyrecest.filters.von_mises_filter import VonMisesFilter


class TestVonMisesFilter(unittest.TestCase):
    def setUp(self):
        self.curr_filter = VonMisesFilter()
        self.vm_prior = VonMisesDistribution(2.1, 1.3)

    def test_setting_state(self):
        vm = self.vm_prior
        self.curr_filter.filter_state = vm
        vm1 = self.curr_filter.filter_state
        self.assertIsInstance(vm1, VonMisesDistribution)
        self.assertEqual(vm.mu, vm1.mu)
        self.assertEqual(vm.kappa, vm1.kappa)

    def test_prediction(self):
        sysnoise = VonMisesDistribution(0, 0.3)

        self.curr_filter.filter_state = self.vm_prior
        self.curr_filter.predict_identity(sysnoise)
        self.assertIsInstance(self.curr_filter.filter_state, VonMisesDistribution)
        self.assertEqual(self.curr_filter.filter_state.mu, 2.1)
        self.assertLess(self.curr_filter.filter_state.kappa, 1.3)

    def test_update(self):
        meas_noise = VonMisesDistribution(0.0, 1.3)
        meas = array(1.1)

        self.curr_filter.filter_state = self.vm_prior
        self.curr_filter.update_identity(meas_noise, meas)
        self.assertIsInstance(self.curr_filter.filter_state, VonMisesDistribution)
        npt.assert_allclose(
            self.curr_filter.get_point_estimate(), (self.vm_prior.mu + meas) / 2.0
        )
        self.assertGreater(self.curr_filter.filter_state.kappa, 1.3)


if __name__ == "__main__":
    unittest.main()
