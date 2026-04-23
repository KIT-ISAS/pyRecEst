import unittest

from pyrecest.distributions import (
    AbstractGridDistribution,
    AbstractPeriodicDistribution,
    AbstractPeriodicGridDistribution,
)


class TestAbstractPeriodicGridDistribution(unittest.TestCase):
    def test_inheritance(self):
        self.assertTrue(
            issubclass(AbstractPeriodicGridDistribution, AbstractGridDistribution)
        )
        self.assertTrue(
            issubclass(AbstractPeriodicGridDistribution, AbstractPeriodicDistribution)
        )


if __name__ == "__main__":
    unittest.main()
