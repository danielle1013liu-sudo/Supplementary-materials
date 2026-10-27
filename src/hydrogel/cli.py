"""Command-line entry points for model comparison, case analysis and prediction."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .cases import evaluate_p32
from .data import ROOT, load_c4, load_config, load_dataset, load_schema, load_split
from .data import validate_features
from .evaluation import evaluate_configuration, summarize_metrics
from .verification import verify_run


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_models(result, destination: Path, root: Path) -> None:
    row = result.metrics.iloc[0]
    destination.mkdir(parents=True)
    entries = []
    for seed, model in result.models.items():
        path = destination / f"model_{seed}.joblib"
        joblib.dump(model, path, compress=3)
        entries.append({"seed": int(seed), "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_json(destination / "manifest.json", {
        "target": row.target, "family": row.family, "imputer": row.imputer,
        "fit_scope": "P32 whole-study exclusion" if row.scope == "P32" else "Training",
        "features": load_schema(root)["predictors"], "models": entries,
    })


def _run_models(args) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    targets = ["tnf", "il10"] if args.target == "all" else [args.target]
    metrics, predictions = [], []
    for target in targets:
        config = load_config(target, args.root)
        if args.command == "cases":
            configurations = [(config["selected_model"], None)]
        elif args.command == "train":
            configurations = [(config["selected_model"], None)]
        else:
            configurations = [
                (family, imputer)
                for family in config["models"]
                for imputer in (config["imputation_methods"] if args.all_imputers else [None])
            ]
        for family, imputer in configurations:
            if args.command == "cases":
                result = evaluate_p32(target, root=args.root)
            else:
                result = evaluate_configuration(target, family, imputer, root=args.root)
            actual_imputer = result.metrics.iloc[0].imputer
            print(f"{target}: {family}, {actual_imputer}", flush=True)
            metrics.append(result.metrics)
            predictions.append(result.predictions)
            if args.command in {"train", "cases"} or args.save_models:
                _save_models(result, args.output / "models" / target / family / actual_imputer, args.root)
    metrics = pd.concat(metrics, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(args.output / "metrics.csv", index=False, float_format="%.17g")
    predictions.to_csv(args.output / "predictions.csv", index=False, float_format="%.17g")
    summarize_metrics(metrics).to_csv(args.output / "summary.csv", index=False, float_format="%.17g")
    if args.check_reference:
        report = verify_run(args.output, args.root)
        _write_json(args.output / "verification.json", report)
        print(json.dumps(report, indent=2))


def _predict(args) -> None:
    manifest = json.loads((args.models / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input, keep_default_na=False, na_values=["", "\u2014"])
    if "target" in frame:
        frame = frame.loc[frame["target"].eq(manifest["target"])].reset_index(drop=True)
    if frame.empty:
        raise ValueError("The input contains no rows for the model target.")
    features = validate_features(frame, load_schema(args.root))
    if list(manifest["features"]) != load_schema(args.root)["predictors"]:
        raise ValueError("The saved model feature order differs from the project schema.")
    predictions = []
    for item in manifest["models"]:
        path = args.models / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"The saved model checksum does not match: {path.name}")
        # Model bundles are produced by the train and cases commands in this project.
        model = joblib.load(path)
        predictions.append(model.predict(features[manifest["features"]]))
    matrix = np.asarray(predictions, dtype=float)
    if matrix.shape[0] != 3 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite predictions from three saved model realisations.")
    identifiers = frame.get("record_id", frame.get("case_id", pd.Series(np.arange(len(frame)))))
    output = pd.DataFrame({
        "record_id": identifiers,
        "prediction_mean_pg_ml": matrix.mean(axis=0),
        "prediction_sample_SD_pg_ml": matrix.std(axis=0, ddof=1),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        output.to_csv(handle, index=False, float_format="%.17g")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository directory containing data and config.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in [
        ("compare", "Compare all four model families using their recorded configurations."),
        ("train", "Fit the selected model and save its three realisations."),
        ("cases", "Fit the selected model with the complete P32 source study excluded."),
    ]:
        sub = commands.add_parser(name, help=description, description=description)
        sub.add_argument("--target", choices=["tnf", "il10", "all"], default="all")
        sub.add_argument("--output", type=Path, required=True, help="New output directory.")
        sub.add_argument("--check-reference", action="store_true")
        sub.add_argument("--save-models", action="store_true", help="Also save fitted model bundles.")
        if name == "compare":
            sub.add_argument("--all-imputers", action="store_true", help="Evaluate all five recorded imputation methods.")
    prediction = commands.add_parser("predict", help="Predict from a saved three-model bundle.")
    prediction.add_argument("--models", type=Path, required=True)
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    verification = commands.add_parser("verify", help="Check an existing run against its numerical reference.")
    verification.add_argument("--run", type=Path, required=True)
    commands.add_parser("validate-data", help="Verify source checksums, categories and recorded partitions.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    with threadpool_limits(limits=1):
        if args.command in {"compare", "train", "cases"}:
            _run_models(args)
        elif args.command == "predict":
            _predict(args)
        elif args.command == "verify":
            print(json.dumps(verify_run(args.run, args.root), indent=2))
        else:
            for target in ["tnf", "il10"]:
                frame = load_dataset(target, args.root)
                split = load_split(target, args.root)
                load_c4(target, args.root)
                print(f"{target}: {len(frame)} rows; {len(split['train_ids'])} Training, {len(split['test_ids'])} Test")
