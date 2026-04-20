import unittest

import numpy as np

from pyrecest.utils import LogisticPairwiseAssociationModel


class TestLogisticPairwiseAssociationModel(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        positive_examples = np.column_stack(
            [
                self.rng.normal(loc=0.2, scale=0.08, size=80),
                self.rng.normal(loc=0.85, scale=0.06, size=80),
            ]
        )
        negative_examples = np.column_stack(
            [
                self.rng.normal(loc=2.4, scale=0.25, size=80),
                self.rng.normal(loc=0.12, scale=0.05, size=80),
            ]
        )
        self.training_features = np.vstack([positive_examples, negative_examples])
        self.training_labels = np.concatenate(
            [np.ones(len(positive_examples), dtype=int), np.zeros(len(negative_examples), dtype=int)]
        )

    def test_fit_separates_match_and_nonmatch_examples(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")
        model.fit(self.training_features, self.training_labels)

        positive_probabilities = model.predict_match_probability(self.training_features[:80])
        negative_probabilities = model.predict_match_probability(self.training_features[80:])

        self.assertTrue(model.converged_)
        self.assertGreater(np.mean(positive_probabilities), 0.95)
        self.assertLess(np.mean(negative_probabilities), 0.05)
        self.assertLess(np.max(negative_probabilities), np.min(positive_probabilities))

    def test_pairwise_cost_matrix_prefers_more_match_like_pairs(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")
        model.fit(self.training_features, self.training_labels)

        pairwise_features = np.array(
            [
                [[0.15, 0.92], [2.2, 0.15]],
                [[2.6, 0.08], [0.18, 0.88]],
            ]
        )

        costs = model.pairwise_cost_matrix(pairwise_features)
        probabilities = model.predict_match_probability(pairwise_features)

        self.assertEqual(costs.shape, (2, 2))
        np.testing.assert_array_equal(np.argmin(costs, axis=1), np.array([0, 1]))
        np.testing.assert_array_equal(np.argmax(probabilities, axis=1), np.array([0, 1]))
        self.assertLess(costs[0, 0], costs[0, 1])
        self.assertLess(costs[1, 1], costs[1, 0])

    def test_flattened_pairwise_training_tensor_is_supported(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")

        training_tensor = np.array(
            [
                [[0.1, 0.95], [2.1, 0.10], [2.4, 0.12]],
                [[2.3, 0.06], [0.2, 0.90], [2.0, 0.20]],
            ]
        )
        label_matrix = np.array([[1, 0, 0], [0, 1, 0]])

        model.fit(training_tensor, label_matrix)

        probabilities = model.predict_match_probability(training_tensor)
        np.testing.assert_array_equal(np.argmax(probabilities, axis=1), np.array([0, 1]))

    def test_constant_feature_is_handled_during_standardization(self):
        model = LogisticPairwiseAssociationModel(class_weight="balanced")

        constant_augmented_features = np.column_stack(
            [self.training_features, np.ones(self.training_features.shape[0])]
        )
        model.fit(constant_augmented_features, self.training_labels)

        probabilities = model.predict_match_probability(constant_augmented_features)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertGreater(np.mean(probabilities[:80]), 0.9)
        self.assertLess(np.mean(probabilities[80:]), 0.1)

    def test_fit_without_intercept_is_supported(self):
        model = LogisticPairwiseAssociationModel(
            class_weight="balanced", fit_intercept=False
        )
        model.fit(self.training_features, self.training_labels)

        probabilities = model.predict_match_probability(self.training_features)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertGreater(np.mean(probabilities[:80]), 0.9)
        self.assertLess(np.mean(probabilities[80:]), 0.1)

    def test_invalid_labels_raise(self):
        model = LogisticPairwiseAssociationModel()
        with self.assertRaises(ValueError):
            model.fit(self.training_features, np.full(self.training_labels.shape, 2))


if __name__ == "__main__":
    unittest.main()
