"""Construct the four regression families from their released parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import make_preprocessor

MODEL_FAMILIES = ("GBR", "RF", "XGB", "LGBM")


def make_estimator(
    family: str, parameters: Mapping[str, Any], seed: int
) -> BaseEstimator:
    """Construct one reproducible model realisation without fitting it."""
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) <= np.iinfo(np.uint32).max
    ):
        raise ValueError("The model seed must be an unsigned 32-bit integer.")
    if "random_state" in parameters:
        raise ValueError("Pass the model seed separately from estimator parameters.")
    arguments = {**parameters, "random_state": int(seed)}
    if family == "GBR":
        return GradientBoostingRegressor(**arguments)
    if family == "RF":
        return RandomForestRegressor(**{**arguments, "n_jobs": 1})
    if family == "XGB":
        from xgboost import XGBRegressor

        if arguments.get("missing") is None:
            arguments["missing"] = np.nan
        return XGBRegressor(**{**arguments, "n_jobs": 1, "verbosity": 0})
    if family == "LGBM":
        from lightgbm import LGBMRegressor

        arguments.update(
            {
                name: int(seed)
                for name in (
                    "bagging_seed",
                    "feature_fraction_seed",
                    "data_random_seed",
                    "drop_seed",
                )
            }
        )
        return LGBMRegressor(
            **{
                **arguments,
                "n_jobs": 1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
                "device_type": "cpu",
            }
        )
    raise ValueError(f"Unknown model family: {family!r}")


def make_pipeline(
    schema: dict[str, Any],
    family: str,
    imputer: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(schema, imputer)),
            ("model", make_estimator(family, parameters, seed)),
        ]
    )
