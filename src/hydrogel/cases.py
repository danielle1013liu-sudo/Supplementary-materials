"""P32 case predictions from the recorded whole-study-excluded population."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_config, load_dataset, load_p32_scope, load_schema
from .evaluation import (
    IDENTIFIERS, METRICS, PREDICTION_COLUMNS, EvaluationResult, _append_scope,
)
from .models import make_pipeline

P32_DOI = "10.1186/s12951-024-02810-z"
P32_REFERENCE_COUNTS = {"tnf": 148, "il10": 129}
P32_CASE_IDS = {
    "tnf": ["SI_TNF_075", "SI_TNF_076"],
    "il10": ["SI_IL10_080", "SI_IL10_081"],
}


def evaluate_p32(target: str, root: str | Path | None = None) -> EvaluationResult:
    """Fit the selected configuration on the saved P32 reference population.

    This population contains all current target records except the complete P32
    source study. Its membership is separate from the primary 80/20 partition.
    The raw GelMA control modulus remains missing and is processed by each
    pipeline's fitted imputer.
    """
    target = target.lower()
    if target not in P32_REFERENCE_COUNTS:
        raise ValueError(f"Unknown target: {target!r}.")
    scope = load_p32_scope(target, root=root)
    configuration = load_config(target, root=root)
    schema = load_schema(root=root)
    frame = load_dataset(target, root=root)
    is_p32 = frame.source_doi.str.strip().str.lower().eq(P32_DOI)
    expected_fit = frame.loc[~is_p32, "record_id"].tolist()
    expected_cases = frame.loc[is_p32, "record_id"].tolist()
    fit_ids = scope["fit_ids"]
    case_ids = scope.get("case_ids", scope["excluded_ids"])
    if scope["excluded_doi"].strip().lower() != P32_DOI:
        raise ValueError("The P32 scope records a different source DOI.")
    if fit_ids != expected_fit or len(fit_ids) != P32_REFERENCE_COUNTS[target]:
        raise ValueError("The saved P32 reference membership differs from the full-study exclusion.")
    if case_ids != expected_cases or case_ids != P32_CASE_IDS[target]:
        raise ValueError("The saved P32 case identities or their order have changed.")
    if scope["excluded_ids"] != expected_cases or set(fit_ids) & set(case_ids):
        raise ValueError("P32 case observations occur in the fitting population.")
    indexed = frame.set_index("record_id", drop=False)
    training = indexed.loc[fit_ids].copy()
    cases = indexed.loc[case_ids].copy()
    if not pd.isna(cases.iloc[0]["Gprime_kPa"]):
        raise ValueError("The raw GelMA control modulus must remain missing.")

    family = configuration["selected_model"]
    specification = configuration["models"][family]
    method = specification["imputer"]
    seeds = configuration["model_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("The case analysis requires three distinct model seeds.")
    features = list(schema["predictors"])
    observed = cases.cytokine_pg_ml.to_numpy(dtype=float)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fitted = {}
    for seed in seeds:
        pipeline = make_pipeline(schema, family, method, specification["parameters"], seed)
        pipeline.fit(training[features], training.cytokine_pg_ml.to_numpy(dtype=float))
        prediction = np.asarray(pipeline.predict(cases[features]), dtype=float)
        identity = {"target": target, "family": family, "imputer": method, "model_seed": int(seed)}
        _append_scope(metric_rows, prediction_rows, identity, "P32", case_ids, observed, prediction)
        fitted[int(seed)] = pipeline
    return EvaluationResult(
        pd.DataFrame(metric_rows, columns=IDENTIFIERS + METRICS),
        pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        fitted,
    )
