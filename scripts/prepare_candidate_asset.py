"""Losslessly pack an approved candidate CSV; do not generate or predict profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(package: Path, output: Path, schema_path: Path) -> dict:
    final = package / "pareto/final"
    provenance = json.loads((final / "PARETO_MANIFEST_FLOAT32.json").read_text(encoding="utf-8"))
    source_paths = [
        "results/ALL_CANDIDATES_AND_PREDICTIONS.csv", "results/PARETO_SUMMARY.json",
        "results/FINAL_PARETO_PROFILES.csv", "results/FINAL_PARETO_REPRESENTATIVES.csv",
        "inputs/feature_schema_v4p2.json",
    ]
    for relative in source_paths:
        if sha256(final / relative) != provenance["artifact_sha256"].get(relative):
            raise ValueError(f"An approved candidate source checksum differs: {relative}.")
    source = final / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv"
    source_hash = sha256(source)
    if source_hash != provenance["artifact_sha256"]["results/ALL_CANDIDATES_AND_PREDICTIONS.csv"]:
        raise ValueError("The candidate CSV does not match the approved Pareto manifest.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    frozen_schema = json.loads((final / "inputs/feature_schema_v4p2.json").read_text(encoding="utf-8"))
    if schema != frozen_schema:
        raise ValueError("The application and candidate schemas differ.")
    data = pd.read_csv(source, float_precision="round_trip", low_memory=False,
                       keep_default_na=False, na_values=[""])
    if data.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidate identifiers in the approved library.")
    summary = json.loads((final / "results/PARETO_SUMMARY.json").read_text(encoding="utf-8"))
    profiles = pd.read_csv(final / "results/FINAL_PARETO_PROFILES.csv", float_precision="round_trip")
    representatives = pd.read_csv(final / "results/FINAL_PARETO_REPRESENTATIVES.csv", float_precision="round_trip")
    counts = {"generated_n": len(data), "predicted_n": int(data.prediction_evaluated.sum()),
              "eligible_n": int(data.eligible.sum()),
              "mean_front_profiles": int(summary["mean_front_profiles"]),
              "final_profiles": len(profiles)}
    if any(counts[name] != summary[name] for name in counts):
        raise ValueError("Recorded candidate counts do not agree with the source summary.")
    arrays, specifications = {}, {}
    for name in data.columns:
        series = data[name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            arrays[name] = series.to_numpy()
            specifications[name] = {"encoding": "direct", "dtype": arrays[name].dtype.str}
        else:
            values = series.fillna("").to_numpy(dtype=str).astype("S")
            levels, codes = np.unique(values, return_inverse=True)
            if len(levels) < len(values) // 4:
                dtype = np.uint8 if len(levels) <= 256 else np.uint16 if len(levels) <= 65536 else np.uint32
                arrays[name] = codes.astype(dtype)
                arrays[name + "__levels"] = levels
                specifications[name] = {"encoding": "dictionary", "dtype": levels.dtype.str}
            else:
                arrays[name] = values
                specifications[name] = {"encoding": "direct", "dtype": values.dtype.str}
    output.mkdir(parents=True, exist_ok=True)
    asset = output / "candidates.npz"
    np.savez_compressed(asset, **arrays)
    # Compare every recovered column with its parsed full-precision source, including
    # excluded rows and provenance-only fields that the application does not load.
    with np.load(asset, allow_pickle=False) as recovered:
        for name, specification in specifications.items():
            actual = recovered[name]
            if specification["encoding"] == "dictionary":
                actual = recovered[name + "__levels"][actual]
            if actual.dtype.kind == "S":
                expected = data[name].fillna("").to_numpy(dtype=str).astype("S")
            else:
                expected = data[name].to_numpy()
            np.testing.assert_array_equal(actual, expected)
    manifest = {
        "format": "hydrogel-candidate-columns-v1", "asset": asset.name,
        "asset_sha256": sha256(asset), "asset_bytes": asset.stat().st_size,
        "source_csv": "pareto/final/results/ALL_CANDIDATES_AND_PREDICTIONS.csv",
        "source_csv_sha256": source_hash, "source_csv_bytes": source.stat().st_size,
        "source_package": package.name, "schema_sha256": sha256(schema_path),
        "features": schema["predictors"], "counts": counts,
        "columns": specifications,
        "member_prediction_columns": {
            target: [name for name in data.columns if re.fullmatch(target + r"_seed_\d+_pg_ml", name)]
            for target in ("tnf", "il10")},
        "canonical_retained_candidate_ids": profiles.candidate_id.tolist(),
        "canonical_representatives": representatives[["display_label", "candidate_id"]].to_dict("records"),
        "selection": {"minimise": "tnf_mean_pg_ml", "maximise": "il10_mean_pg_ml",
                      "minimum_pairing_fronts": 6, "total_pairings": 9,
                      "ties": "Preserve distinct profiles with exactly equal objective values; no tolerance.",
                      "eligibility": "Both AD ratios <= 1 and category support in both Training populations; all six recorded predictions non-negative.",
                      "universe": "The same complete eligible subset for the mean front and every target-model pairing."},
        "source_verification": {"all_columns_losslessly_recovered": True,
                                "source_artifact_sha256": {name: provenance["artifact_sha256"][name] for name in source_paths},
                                "candidate_generation_performed": False,
                                "model_inference_performed": False,
                                "source_manifest_sha256": sha256(final / "PARETO_MANIFEST_FLOAT32.json"),
                                "pareto_implementation_sha256": sha256(package / "scripts/pareto/pareto_math.py")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return {"counts": counts, "asset_bytes": asset.stat().st_size, "all_columns_lossless": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(prepare(options.package, options.output, options.schema), indent=2))
