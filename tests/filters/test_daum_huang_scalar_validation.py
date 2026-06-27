import unittest

import numpy as np

from pyrecest.filters import EDHParticleFlowFilter


class DaumHuangScalarValidationTest(unittest.TestCase):
    def test_n_steps_rejects_numpy_boolean(self):
        with self.assertRaisesRegex(ValueError, "n_steps must be a positive integer"):
            EDHParticleFlowFilter(
                n_particles=2,
                dim=1,
                n_steps=np.bool_(True),
            )

    def test_jitter_rejects_boolean(self):
        for bad_value in (True, np.bool_(False)):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "jitter must be finite and nonnegative",
                ):
                    EDHParticleFlowFilter(
                        n_particles=2,
                        dim=1,
                        jitter=bad_value,
                    )


if __name__ == "__main__":
    unittest.main()
