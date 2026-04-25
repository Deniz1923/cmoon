"""Pickle-stable ML ensemble estimator."""
from __future__ import annotations

import numpy as np


class EnsembleModel:
    """Average predicted class probabilities from two fitted estimators."""

    def __init__(self, rf: object, boost: object):
        self.rf = rf
        self.boost = boost
        self.classes_ = self._classes()

    def predict_proba(self, X) -> np.ndarray:
        rf_proba = self._aligned_proba(self.rf, X)
        boost_proba = self._aligned_proba(self.boost, X)
        return (rf_proba + boost_proba) / 2.0

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = [
            np.asarray(getattr(model, "feature_importances_"), dtype=float)
            for model in (self.rf, self.boost)
            if hasattr(model, "feature_importances_")
        ]
        if not importances:
            raise AttributeError("No wrapped estimator exposes feature_importances_")
        return np.mean(importances, axis=0)

    def _classes(self) -> np.ndarray:
        for model in (self.rf, self.boost):
            classes = getattr(model, "classes_", None)
            if classes is not None:
                return np.asarray(classes)
        return np.asarray([0, 1])

    def _aligned_proba(self, model: object, X) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"{type(model).__name__} does not expose predict_proba")

        proba = np.asarray(model.predict_proba(X), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", self.classes_))

        if np.array_equal(model_classes, self.classes_):
            return proba

        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=float)
        for source_col, klass in enumerate(model_classes):
            matches = np.where(self.classes_ == klass)[0]
            if len(matches):
                aligned[:, matches[0]] = proba[:, source_col]
        return aligned
