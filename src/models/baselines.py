#!/usr/bin/env python3
"""
baselines.py - Autoregressive reference rules and supervised classifiers.

Autoregressive baselines (no learning beyond the history features):
  PreviousOutcomeBaseline     score = outcome of the most recently finished build
  RollingFailureRateBaseline  score = mean outcome of the last five finished builds
  HistoryLogisticRegression   logistic regression on the history feature group only

Supervised classifiers on the full feature matrix:
  RandomForestBaseline        scikit-learn RandomForestClassifier
  GradientBoostingBaseline    scikit-learn HistGradientBoostingClassifier

All hyperparameters are fixed a priori and listed in the class constructors.
The gradient boosting implementation is HistGradientBoostingClassifier.
"""

from typing import List

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class _ColumnRule:
    """A scorer that reads named columns from the feature matrix."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)

    def _col(self, name: str) -> int:
        return self.feature_names.index(name)

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) >= 0.5).astype(int)


class PreviousOutcomeBaseline(_ColumnRule):
    """Outcome of the most recently finished build, with the five build rolling
    rate as a tie breaker so the score is a proper ranking."""

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        prev = X[:, self._col("prev_finished_outcome")]
        rate = X[:, self._col("hist_failure_rate_w5")]
        return prev + 1e-3 * rate


class RollingFailureRateBaseline(_ColumnRule):
    def __init__(self, feature_names: List[str], window: int = 5):
        super().__init__(feature_names)
        self.window = window

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        rate = X[:, self._col(f"hist_failure_rate_w{self.window}")]
        prev = X[:, self._col("prev_finished_outcome")]
        return rate + 1e-3 * prev


class HistoryLogisticRegression:
    """Logistic regression restricted to the history feature group."""

    def __init__(self, feature_names: List[str], history_features: List[str], C: float = 1.0, random_state: int = 42):
        self.cols = [feature_names.index(f) for f in history_features if f in feature_names]
        self.model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X[:, self.cols], y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X[:, self.cols])[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X[:, self.cols])


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 150, max_depth: int = 10, min_samples_leaf: int = 5, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class GradientBoostingBaseline:
    def __init__(self, n_estimators: int = 150, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight="balanced",
            early_stopping=False,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LogisticRegressionBaseline:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
