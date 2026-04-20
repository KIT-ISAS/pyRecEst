"""Feature-based probabilistic association models.

This module adds a lightweight learned association model that is useful for
cross-session neuron identity tracking. In settings such as Track2p or CaliAli,
pairwise association decisions are usually based on several cues at once, for
example registered centroid distance, ROI overlap, mask correlation, or trace
similarity. The :class:`LogisticPairwiseAssociationModel` learns how to fuse
those cues into calibrated match probabilities and assignment costs.

The API accepts either a standard design matrix with shape
``(n_samples, n_features)`` or a higher-order pairwise feature tensor with shape
``(..., n_features)`` together with a label array of shape ``(...)``. This is
convenient when ground-truth match matrices are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


_COST_MODE = Literal["negative_log_probability", "one_minus_probability"]


@dataclass(frozen=True)
class _BinaryClassWeights:
    """Resolved weights for the negative and positive class."""

    negative: float
    positive: float

    def as_array(self, labels: np.ndarray) -> np.ndarray:
        return np.where(labels == 1, self.positive, self.negative).astype(float)


class LogisticPairwiseAssociationModel:
    """Learn pairwise match probabilities from arbitrary association features.

    The model uses binary logistic regression with optional feature
    standardization, L2 regularization, and support for severe class imbalance.
    It is intentionally dependency-light and implemented with NumPy only so that
    it can be used inside PyRecEst without adding a machine-learning stack.

    Parameters
    ----------
    fit_intercept:
        Whether to learn an intercept term.
    l2_regularization:
        Non-negative L2 regularization strength applied to the coefficients.
        The intercept is not regularized.
    max_iterations:
        Maximum number of Newton / IRLS iterations.
    tolerance:
        Stop once the largest absolute parameter update falls below this value.
    standardize:
        Whether to z-score features before optimization.
    class_weight:
        ``None`` for unweighted fitting, ``"balanced"`` to automatically rebalance
        positive and negative examples, or a dictionary containing weights for
        labels ``0`` and ``1``.
    probability_clip:
        Numerical clipping applied to probabilities before taking logarithms.

    Notes
    -----
    The fit method accepts both a flattened feature matrix and a pairwise tensor.
    This makes it convenient to train directly from a ground-truth match matrix:

    >>> model = LogisticPairwiseAssociationModel(class_weight="balanced")
    >>> model.fit(pairwise_features, match_labels)
    >>> costs = model.pairwise_cost_matrix(pairwise_features)

    Here ``pairwise_features`` may contain, for example, centroid distance,
    overlap, and shape-correlation channels for every candidate neuron pair.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        l2_regularization: float = 1.0e-3,
        max_iterations: int = 100,
        tolerance: float = 1.0e-8,
        standardize: bool = True,
        class_weight: str | dict[int, float] | None = "balanced",
        probability_clip: float = 1.0e-12,
    ) -> None:
        if l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if not 0.0 < probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, 0.5)")
        if class_weight != "balanced" and class_weight is not None and not isinstance(class_weight, dict):
            raise ValueError(
                "class_weight must be None, 'balanced', or a dictionary with labels 0 and 1"
            )

        self.fit_intercept = fit_intercept
        self.l2_regularization = float(l2_regularization)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.standardize = bool(standardize)
        self.class_weight = class_weight
        self.probability_clip = float(probability_clip)

        self.n_features_in_: int | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.class_weights_: _BinaryClassWeights | None = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        values = np.asarray(values, dtype=float)
        result = np.empty_like(values, dtype=float)
        nonnegative = values >= 0.0
        result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
        exp_values = np.exp(values[~nonnegative])
        result[~nonnegative] = exp_values / (1.0 + exp_values)
        return result

    @staticmethod
    def _ensure_binary_labels(labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.size == 0:
            raise ValueError("At least one labeled example is required")

        unique_labels = np.unique(labels)
        if not np.all(np.isin(unique_labels, [0, 1, False, True])):
            raise ValueError("labels must only contain binary values 0/1 or False/True")

        labels = labels.astype(int)
        if np.unique(labels).size != 2:
            raise ValueError("Both negative and positive examples are required")
        return labels

    @staticmethod
    def _prepare_training_features(features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features[:, None]
        if features.ndim < 2:
            raise ValueError("features must be at least one-dimensional")
        flattened = features.reshape(-1, features.shape[-1])
        if flattened.shape[0] == 0:
            raise ValueError("At least one feature vector is required")
        if not np.all(np.isfinite(flattened)):
            raise ValueError("features must be finite")
        return flattened

    @staticmethod
    def _prepare_prediction_features(
        features: np.ndarray,
        expected_feature_dimension: int,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            if features.shape[0] != expected_feature_dimension:
                raise ValueError(
                    "A one-dimensional prediction input must contain exactly one feature vector"
                )
            original_shape = ()
            flattened = features[None, :]
        else:
            if features.shape[-1] != expected_feature_dimension:
                raise ValueError(
                    "The last axis of features must equal the number of fitted features"
                )
            original_shape = features.shape[:-1]
            flattened = features.reshape(-1, expected_feature_dimension)

        if not np.all(np.isfinite(flattened)):
            raise ValueError("features must be finite")
        return flattened, original_shape

    def _resolve_class_weights(self, labels: np.ndarray) -> _BinaryClassWeights:
        if self.class_weight is None:
            return _BinaryClassWeights(negative=1.0, positive=1.0)

        if self.class_weight == "balanced":
            class_counts = np.bincount(labels, minlength=2).astype(float)
            if np.any(class_counts == 0.0):
                raise ValueError("Both classes must be present when using balanced class weights")
            total_count = class_counts.sum()
            return _BinaryClassWeights(
                negative=total_count / (2.0 * class_counts[0]),
                positive=total_count / (2.0 * class_counts[1]),
            )

        if 0 not in self.class_weight or 1 not in self.class_weight:
            raise ValueError("class_weight dictionaries must contain entries for both 0 and 1")
        negative_weight = float(self.class_weight[0])
        positive_weight = float(self.class_weight[1])
        if negative_weight <= 0.0 or positive_weight <= 0.0:
            raise ValueError("class weights must be positive")
        return _BinaryClassWeights(negative=negative_weight, positive=positive_weight)

    def _build_effective_sample_weights(
        self,
        labels: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> np.ndarray:
        weights = np.ones(labels.shape[0], dtype=float)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != labels.shape:
                raise ValueError("sample_weight must match the flattened label shape")
            if np.any(sample_weight < 0.0):
                raise ValueError("sample_weight must be non-negative")
            weights *= sample_weight

        class_weights = self._resolve_class_weights(labels)
        self.class_weights_ = class_weights
        weights *= class_weights.as_array(labels)

        if not np.any(weights > 0.0):
            raise ValueError("At least one example must receive positive weight")
        return weights

    def _fit_standardization(self, features: np.ndarray) -> np.ndarray:
        if self.standardize:
            feature_mean = features.mean(axis=0)
            feature_scale = features.std(axis=0)
            feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
        else:
            feature_mean = np.zeros(features.shape[1], dtype=float)
            feature_scale = np.ones(features.shape[1], dtype=float)

        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        return (features - self.feature_mean_) / self.feature_scale_

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("The association model is not fitted")
        return (features - self.feature_mean_) / self.feature_scale_

    def _design_matrix(self, features: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return np.concatenate(
                [np.ones((features.shape[0], 1), dtype=float), features], axis=1
            )
        return features

    def _require_fitted(self) -> None:
        if self.n_features_in_ is None or self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("The association model must be fitted before use")

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "LogisticPairwiseAssociationModel":
        """Fit the association model.

        Parameters
        ----------
        features:
            Feature matrix of shape ``(n_samples, n_features)`` or feature tensor
            of shape ``(..., n_features)``.
        labels:
            Binary labels with shape ``(n_samples,)`` or ``(...)`` matching the
            leading dimensions of ``features``.
        sample_weight:
            Optional non-negative per-example weights with the same shape as
            ``labels``.
        """
        flattened_features = self._prepare_training_features(features)
        labels = self._ensure_binary_labels(np.asarray(labels).reshape(-1))

        if flattened_features.shape[0] != labels.shape[0]:
            raise ValueError(
                "The number of feature vectors must equal the number of labels after flattening"
            )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sample_weight.shape[0] != labels.shape[0]:
                raise ValueError("sample_weight must match the number of flattened labels")

        self.n_features_in_ = flattened_features.shape[1]
        standardized_features = self._fit_standardization(flattened_features)
        design_matrix = self._design_matrix(standardized_features)
        effective_weights = self._build_effective_sample_weights(labels, sample_weight)

        parameter_vector = np.zeros(design_matrix.shape[1], dtype=float)
        regularization_mask = np.ones_like(parameter_vector)
        if self.fit_intercept:
            regularization_mask[0] = 0.0

        self.converged_ = False
        for iteration in range(1, self.max_iterations + 1):
            logits = design_matrix @ parameter_vector
            probabilities = self._sigmoid(logits)
            residual = (probabilities - labels) * effective_weights
            gradient = design_matrix.T @ residual
            if self.l2_regularization > 0.0:
                gradient += self.l2_regularization * regularization_mask * parameter_vector

            curvature = effective_weights * probabilities * (1.0 - probabilities)
            hessian = design_matrix.T @ (design_matrix * curvature[:, None])
            if self.l2_regularization > 0.0:
                hessian += self.l2_regularization * np.diag(regularization_mask)

            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient

            parameter_vector -= step
            self.n_iter_ = iteration
            if np.max(np.abs(step)) <= self.tolerance:
                self.converged_ = True
                break

        if self.fit_intercept:
            self.intercept_ = float(parameter_vector[0])
            self.coefficients_ = parameter_vector[1:]
        else:
            self.intercept_ = 0.0
            self.coefficients_ = parameter_vector

        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return posterior log-odds for the provided feature vectors."""
        self._require_fitted()
        assert self.n_features_in_ is not None
        flattened_features, original_shape = self._prepare_prediction_features(
            features, self.n_features_in_
        )
        standardized_features = self._transform_features(flattened_features)
        if self.fit_intercept:
            parameter_vector = np.concatenate(([self.intercept_], self.coefficients_))
        else:
            parameter_vector = self.coefficients_
        logits = self._design_matrix(standardized_features) @ parameter_vector
        if original_shape == ():
            return np.asarray(logits[0])
        return logits.reshape(original_shape)

    def predict_log_odds(self, features: np.ndarray) -> np.ndarray:
        """Alias for :meth:`decision_function`."""
        return self.decision_function(features)

    def predict_match_probability(self, features: np.ndarray) -> np.ndarray:
        """Return posterior match probabilities for the provided feature vectors."""
        log_odds = self.decision_function(features)
        return self._sigmoid(np.asarray(log_odds, dtype=float))

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match decisions using the supplied threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        probabilities = self.predict_match_probability(features)
        return probabilities >= threshold

    def pairwise_cost_matrix(
        self,
        pairwise_features: np.ndarray,
        *,
        mode: _COST_MODE = "negative_log_probability",
    ) -> np.ndarray:
        """Convert pairwise features into an assignment cost matrix.

        Parameters
        ----------
        pairwise_features:
            Pairwise feature tensor of shape ``(..., n_features)``. For neuron
            tracking this is typically ``(n_reference_rois, n_candidate_rois,
            n_features)``.
        mode:
            ``"negative_log_probability"`` returns ``-log p(match | features)``.
            ``"one_minus_probability"`` returns ``1 - p(match | features)``.
        """
        probabilities = np.clip(
            self.predict_match_probability(pairwise_features),
            self.probability_clip,
            1.0 - self.probability_clip,
        )
        if mode == "negative_log_probability":
            return -np.log(probabilities)
        if mode == "one_minus_probability":
            return 1.0 - probabilities
        raise ValueError(f"Unsupported cost mode: {mode}")
