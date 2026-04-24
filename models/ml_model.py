from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features.engineer import FEATURE_COLUMNS, make_supervised_frame
from models.base_model import BaseSignalModel, RawSignal, bounded_confidence, direction_from_score


class NumpyRidgeModel(BaseSignalModel):
    """Small deterministic ML baseline with no sklearn runtime dependency."""

    name = "numpy_ridge"

    def __init__(
        self,
        *,
        ridge_alpha: float = 1.0,
        threshold: float = 0.03,
        horizon: int = 1,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.threshold = threshold
        self.horizon = horizon
        self.coefficients: dict[str, np.ndarray] = {}
        self.feature_mean: dict[str, np.ndarray] = {}
        self.feature_scale: dict[str, np.ndarray] = {}

    def train(self, asset_frames: dict[str, pd.DataFrame]) -> None:
        for asset, frame in asset_frames.items():
            supervised = make_supervised_frame(frame, horizon=self.horizon)
            if len(supervised) < len(FEATURE_COLUMNS) + 5:
                continue

            x = supervised.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
            y = supervised["future_return"].to_numpy(dtype=float)
            mean = x.mean(axis=0)
            scale = x.std(axis=0)
            scale[scale == 0.0] = 1.0
            x_scaled = (x - mean) / scale
            x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

            penalty = np.eye(x_design.shape[1]) * self.ridge_alpha
            penalty[0, 0] = 0.0
            beta = np.linalg.pinv(x_design.T @ x_design + penalty) @ x_design.T @ y

            self.coefficients[asset] = beta
            self.feature_mean[asset] = mean
            self.feature_scale[asset] = scale

    def predict(self, features: pd.DataFrame, *, asset: str) -> RawSignal:
        if features.empty or asset not in self.coefficients:
            return RawSignal(asset=asset, score=0.0, confidence=0.0, direction=0)

        x = features.iloc[-1].reindex(FEATURE_COLUMNS).fillna(0.0).to_numpy(dtype=float)
        x_scaled = (x - self.feature_mean[asset]) / self.feature_scale[asset]
        x_design = np.concatenate([[1.0], x_scaled])
        score = float(x_design @ self.coefficients[asset])

        return RawSignal(
            asset=asset,
            score=score,
            confidence=bounded_confidence(score, scale=0.025),
            direction=direction_from_score(score, self.threshold),
            metadata={"model": self.name},
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyRidgeModel":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded


def train_from_raw(asset_frames: dict[str, pd.DataFrame]) -> NumpyRidgeModel:
    model = NumpyRidgeModel()
    model.train(asset_frames)
    return model
