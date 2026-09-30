"""Shared numerical contracts and full-precision table I/O."""

from pathlib import Path
import hashlib
import json
import pickle
import zlib
import numpy as np
import pandas as pd
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.base import BaseEstimator


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/pareto"
SCHEMA = read(ROOT / "data/feature_schema.json")
FEATURES = SCHEMA["raw_feature_column_order"]
NUM = SCHEMA["numeric_feature_columns_in_recovered_preprocessing_order"]
CAT = SCHEMA["categorical_feature_columns"]
MATERIAL = [
    "polymer_origin_class",
    "material_bioactive_cue_class",
    "gelation_or_crosslinking_mechanism",
    "network_density_proxy",
]
EXPERIMENT = [
    "macrophage_model_class",
    "stimulation_context",
    "cell_material_interaction_mode",
]
SEEDS = {"tnf": [3089, 4280, 5793], "il10": [85, 115, 158]}
KEY = "dataset_row_id"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def csv(path):
    return pd.read_csv(
        path,
        float_precision="round_trip",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )


def table(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def describe(obj):
    if isinstance(obj, BaseEstimator):
        return {
            "__estimator__": obj.__class__.__module__ + "." + obj.__class__.__name__,
            "parameters": {
                k: describe(v) for k, v in obj.get_params(deep=False).items()
            },
        }
    if isinstance(obj, type):
        return {"__type__": obj.__module__ + "." + obj.__name__}
    if isinstance(obj, tuple):
        return {"__tuple__": [describe(v) for v in obj]}
    if isinstance(obj, list):
        return [describe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): describe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return {"__array__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return describe(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return {"__float__": str(obj)}
    return obj


def dataset(target):
    return csv(ROOT / f"data/{target}.csv")


def training(target):
    context = read(DATA / f"{target}_context.json")
    return (
        dataset(target)
        .set_index(KEY, drop=False)
        .loc[context["Training_IDs"]]
        .reset_index(drop=True)
    )


def models(target):
    return [
        (
            seed,
            pickle.loads(
                zlib.decompress((ROOT / f"models/{target}_{seed}.pklz").read_bytes())
            ),
        )
        for seed in SEEDS[target]
    ]


def front(x, y):
    """Minimise x and maximise y, retaining every exact objective tie."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Objectives must be aligned finite arrays")
    mask = np.zeros(len(x), dtype=bool)
    order = np.argsort(x, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(x[order]) != 0) + 1]
    if not len(order):
        return mask
    maxima = np.maximum.reduceat(y[order], starts)
    previous = np.r_[-np.inf, np.maximum.accumulate(maxima)[:-1]]
    for begin, end, maximum, best in zip(
        starts, np.r_[starts[1:], len(order)], maxima, previous
    ):
        if maximum > best:
            group = order[begin:end]
            mask[group[y[group] == maximum]] = True
    return mask
