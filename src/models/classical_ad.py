#!/usr/bin/env python3
"""
classical_ad.py - Isolation forest and PCA reconstruction novelty detectors.

Both are fitted on passing training builds only and produce a continuous
anomaly score in which larger means more anomalous.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class IsolationForestAD:
    def __init__(self, n_estimators: int = 150, contamination: float = 0.10, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )
        self.threshold: float = 0.0

    def fit(self, X_normal: np.ndarray):
        self.model.fit(X_normal)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

    def calibrate_threshold(self, X_val_normal: np.ndarray, percentile: float = 92.0) -> float:
        self.threshold = float(np.percentile(self.predict_score(X_val_normal), percentile))
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) > self.threshold).astype(int)


class PCAReconstructionAD:
    """Linear reconstruction baseline: residual norm after projection onto the
    principal subspace that explains a fixed share of variance."""

    def __init__(self, n_components: float = 0.85, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.threshold: float = 0.0

    def fit(self, X_normal: np.ndarray):
        n_comp = self.n_components
        if isinstance(n_comp, float):
            n_comp = min(n_comp, 0.999)
        self.pca = PCA(n_components=n_comp, random_state=self.random_state, svd_solver="full")
        self.pca.fit(X_normal)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X_proj = self.pca.inverse_transform(self.pca.transform(X))
        return np.mean((X - X_proj) ** 2, axis=1)

    def calibrate_threshold(self, X_val_normal: np.ndarray, percentile: float = 92.0) -> float:
        self.threshold = float(np.percentile(self.predict_score(X_val_normal), percentile))
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) > self.threshold).astype(int)
