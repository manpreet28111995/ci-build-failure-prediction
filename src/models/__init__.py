"""Detector and baseline implementations."""

from .autoencoder import TelemetryAutoencoder
from .classical_ad import IsolationForestAD, PCAReconstructionAD
from .baselines import (
    PreviousOutcomeBaseline,
    RollingFailureRateBaseline,
    HistoryLogisticRegression,
    RandomForestBaseline,
    GradientBoostingBaseline,
    LogisticRegressionBaseline,
)

__all__ = [
    "TelemetryAutoencoder",
    "IsolationForestAD",
    "PCAReconstructionAD",
    "PreviousOutcomeBaseline",
    "RollingFailureRateBaseline",
    "HistoryLogisticRegression",
    "RandomForestBaseline",
    "GradientBoostingBaseline",
    "LogisticRegressionBaseline",
]
