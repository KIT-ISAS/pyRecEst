import unittest
import numpy as np
from VMDistribution import VMDistribution


class TestVMDistribution(unittest.TestCase):
    def test_vm_init(self):
        dist1 = VMDistribution(np.array(0.0, dtype=np.float32), np.array(1.0, dtype=np.float32))
        dist2 = VMDistribution(np.array(2.0, dtype=np.float32), np.array(1.0, dtype=np.float32))
        self.assertEqual(dist1.kappa, dist2.kappa)
        self.assertNotEqual(dist1.mu, dist2.mu)

    def test_pdf(self):
        dist = VMDistribution(np.array(2.0, dtype=np.float32), np.array(1.0, dtype=np.float32))
        xs = np.linspace(1, 7, 7, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            dist.pdf(xs),
            np.array(
                [
                    0.215781465110296,
                    0.341710488623463,
                    0.215781465110296,
                    0.0829150854731715,
                    0.0467106111086458,
                    0.0653867888824553,
                    0.166938593220285,
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
