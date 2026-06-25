import unittest

from pyrecest.backend import array
from pyrecest.utils import CalibratedPairwiseAssociationModel


class TestCalibratedAssociationProbabilityBounds(unittest.TestCase):
    def test_rejects_direct_probabilities_outside_unit_interval(self):
        class DirectProbabilityModel:
            def __init__(self, probabilities):
                self.probabilities = probabilities

            def predict_match_probability(self, features):
                del features
                return array(self.probabilities)

        features = array([[0.1], [0.2]])
        invalid_probability_vectors = (
            [-0.1, 0.5],
            [0.5, 1.1],
        )

        for probabilities in invalid_probability_vectors:
            with self.subTest(probabilities=probabilities):
                model = CalibratedPairwiseAssociationModel(
                    DirectProbabilityModel(probabilities), feature_names=("distance",)
                )

                with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    model.predict_match_probability(features)

    def test_rejects_predict_proba_probabilities_outside_unit_interval(self):
        class PredictProbaModel:
            classes_ = [0, 1]

            def predict_proba(self, features):
                del features
                return array([[0.8, 1.2], [0.9, 0.1]])

        model = CalibratedPairwiseAssociationModel(
            PredictProbaModel(), feature_names=("distance",)
        )

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            model.predict_match_probability(array([[0.1], [0.2]]))

    def test_rejects_negative_cost_derived_probabilities_above_one(self):
        class NegativeCostModel:
            def pairwise_cost_matrix(self, features):
                del features
                return array([-1.0, 0.0])

        model = CalibratedPairwiseAssociationModel(
            NegativeCostModel(), feature_names=("distance",)
        )

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            model.predict_match_probability(array([[0.1], [0.2]]))


if __name__ == "__main__":
    unittest.main()
