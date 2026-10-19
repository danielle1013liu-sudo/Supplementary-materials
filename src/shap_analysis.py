"""Reproduce global and external-case SHAP from the frozen primary models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (ROOT, ID, check_environment, check_inputs, feature_names,
                   load_data, load_cases, split_definition, save_csv, save_json)
from .models import load_models
from .interpretation import (check_references, explain, save_explanation,
                             check_shap_reference, settings)


def run(output):
    output = Path(output)
    check_environment()
    check_inputs()
    check_references()
    importance = []
    for target in ("tnf", "il10"):
        frame = load_data(target)
        split = split_definition(target)
        models = load_models(target)
        membership = ["Training-in-sample" if row in split["Training_IDs"] else "Test-out-of-fit" for row in frame[ID]]
        arrays = explain(models, frame[feature_names()], frame[ID].tolist(), membership,
                         output / "ALL_ROWS_DESCRIPTIVE", target)
        check_shap_reference(target, "ALL_ROWS_DESCRIPTIVE", arrays)
        index = [frame[ID].tolist().index(row) for row in split["Test_IDs"]]
        test = {
            "grouped": arrays["grouped"][:, index],
            "consensus": arrays["grouped"][:, index].mean(axis=0),
            "predictions": arrays["predictions"][:, index],
            "expected_values": arrays["expected_values"],
            "ids": np.asarray(split["Test_IDs"], dtype=str),
        }
        check_shap_reference(target, "TEST_ONLY", test)
        save_explanation(test, ["Test-out-of-fit"] * len(index), output / "TEST_ONLY", target)
        values = load_cases(target)
        external = explain(models, values[feature_names()], ["C4_Group2", "C4_Group4"],
                           ["external"] * 2, output / "C4_EXTERNAL", target)
        check_shap_reference(target, "C4_EXTERNAL", external)
        magnitude = abs(test["consensus"]).mean(axis=0)
        individual = abs(test["grouped"]).mean(axis=1)
        for column, descriptor in enumerate(feature_names()):
            importance.append({
                "target": target, "descriptor": descriptor, "n_Test": len(index),
                "consensus_mean_abs_SHAP_pg_ml": magnitude[column],
                "importance_percent": 100 * magnitude[column] / magnitude.sum(),
                "individual_importance_mean_pg_ml": individual[:, column].mean(),
                "individual_importance_sample_SD_pg_ml": individual[:, column].std(ddof=1),
            })
    save_csv(pd.DataFrame(importance), output / "test_importance.csv")
    save_json({**settings(), "fit_scope": "Primary Training-only saved models",
               "importance_scope": "Test observations only",
               "all_rows_scope": "Descriptive Training and Test explanations; not OOF SHAP",
               "accepted_reference_arrays_equal": True}, output / "settings.json")
    check_inputs()
    check_references()
    return importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/shap")
    args = parser.parse_args()
    run(args.output)
    print("Primary SHAP arrays reproduce all accepted references exactly.")
