"""Numerical imputation and fixed-schema categorical encoding."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

IMPUTATION_METHODS = (
    "median",
    "mean",
    "iterative_bayesian_ridge",
    "scaled_knn_3",
    "scaled_knn_5",
)


class ScaledKNNImputer(TransformerMixin, BaseEstimator):
    """Impute using standardised distances, then restore the original units.

    Both the scaler and neighbour reference are fitted on the supplied training
    rows. Observed values are copied back exactly after the inverse transform.
    """

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    @staticmethod
    def _values(X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not np.isfinite(values[~np.isnan(values)]).all():
            raise ValueError(
                "Expected a two-dimensional array without infinite values."
            )
        return values

    def fit(self, X: Any, y: Any = None) -> ScaledKNNImputer:
        values = self._values(X)
        if np.isnan(values).all(axis=0).any():
            raise ValueError("A numerical training column is entirely missing.")
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.scaler_ = StandardScaler().fit(values)
        self.imputer_ = KNNImputer(
            n_neighbors=self.n_neighbors, weights="distance"
        ).fit(self.scaler_.transform(values))
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["scaler_", "imputer_"])
        values = self._values(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("The number of numerical descriptors has changed.")
        transformed = self.imputer_.transform(self.scaler_.transform(values))
        result = self.scaler_.inverse_transform(transformed)
        observed = ~np.isnan(values)
        result[observed] = values[observed]
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_.copy()
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def make_preprocessor(schema: dict[str, Any], method: str) -> ColumnTransformer:
    """Build an unfitted transformer; numerical descriptors retain their units."""
    if method in ("median", "mean"):
        numeric = SimpleImputer(strategy=method)
    elif method == "iterative_bayesian_ridge":
        numeric = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            sample_posterior=False,
            random_state=42,
        )
    elif method in ("scaled_knn_3", "scaled_knn_5"):
        numeric = ScaledKNNImputer(3 if method == "scaled_knn_3" else 5)
    else:
        raise ValueError(f"Unknown imputation method: {method!r}")
    categorical_names = [
        name for name in schema["predictors"] if name in schema["categorical"]
    ]
    categories = [
        list(schema["categorical"][name]["labels"]) for name in categorical_names
    ]
    categorical = OneHotEncoder(
        categories=categories,
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )
    return ColumnTransformer(
        [
            ("numerical", Pipeline([("imputer", numeric)]), schema["numeric"]),
            ("categorical", categorical, categorical_names),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
