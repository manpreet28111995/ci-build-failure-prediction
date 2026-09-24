#!/usr/bin/env python3
"""
autoencoder.py - Reconstruction based novelty detector for tabular build features.

The detector is a bottleneck multilayer perceptron trained to reconstruct the
scaled features of passing builds. The anomaly score is the mean squared
reconstruction error, weighted per feature by the inverse of the feature's
variance among passing training builds.

The legacy weighting is 1 / (var_j + 1e-4). For a feature that is constant among passing
training builds this weight is 10,000, so any non zero test value of that one
column dominated the score and the detector degenerated into a single column
novelty flag. Two changes remove that failure mode:

  * features whose variance among passing training builds is below 1e-8 are
    excluded from the score altogether (they still enter the network input);
  * the remaining weights are floored at one tenth of the median variance, so
    no single feature can carry an unbounded share of the score.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class TelemetryAutoencoder:
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (32, 16, 8, 16, 32),
        activation: str = "relu",
        alpha: float = 1e-4,
        max_iter: int = 500,
        batch_size: int = 64,
        learning_rate_init: float = 0.001,
        reconstruction_mode: str = "normalized",
        variance_floor_fraction: float = 0.1,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.reconstruction_mode = reconstruction_mode
        self.variance_floor_fraction = variance_floor_fraction
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            learning_rate="adaptive",
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )
        self.threshold: float = 0.0
        self.feature_weights: Optional[np.ndarray] = None
        self.active_features: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X_normal: np.ndarray):
        var = np.var(X_normal, axis=0)
        active = var > 1e-8
        if active.sum() == 0:
            active = np.ones_like(active, dtype=bool)
        median_var = float(np.median(var[active]))
        floor = max(self.variance_floor_fraction * median_var, 1e-8)
        weights = np.zeros_like(var, dtype=float)
        weights[active] = 1.0 / np.maximum(var[active], floor)
        self.feature_weights = weights
        self.active_features = active

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(X_normal, X_normal)
        self.is_fitted = True
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Autoencoder is not fitted")
        return self.model.predict(X)

    def compute_feature_attribution(self, X: np.ndarray) -> np.ndarray:
        diff_sq = (X - self.reconstruct(X)) ** 2
        if self.reconstruction_mode == "normalized":
            return diff_sq * self.feature_weights
        return diff_sq

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        attr = self.compute_feature_attribution(X)
        denom = max(int(self.active_features.sum()), 1) if self.reconstruction_mode == "normalized" else X.shape[1]
        return attr.sum(axis=1) / denom

    def calibrate_threshold(self, X_val_normal: np.ndarray, percentile: float = 92.0) -> float:
        self.threshold = float(np.percentile(self.compute_reconstruction_error(X_val_normal), percentile))
        return self.threshold

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.compute_reconstruction_error(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) > self.threshold).astype(int)
