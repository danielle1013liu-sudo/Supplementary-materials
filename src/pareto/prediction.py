"""Batched inference and saved Training-reference applicability domains."""

from .common import (
    DATA,
    FEATURES,
    Path,
    ROOT,
    SEEDS,
    digest,
    dump,
    joblib,
    models,
    np,
    pd,
    read,
    sha,
    table,
    training,
)
from sklearn.metrics import pairwise_distances
from time import perf_counter


def run(output):
    """Write this numerical stage below the separate output directory."""
    output = Path(output)
    search = output / "search"
    complete = read(search / "GENERATION_COMPLETE.json")
    assert complete["all_ten_descriptors_vary"]
    assert sha(search / "CANDIDATE_INPUTS.csv") == complete["candidate_table_sha256"]
    assert sha(search / "SEARCH_SPACE_COVERAGE.csv") == complete["coverage_sha256"]
    config = read(search / "SEARCH_CONFIGURATION.json")
    assert config["accepted_seeds"] == SEEDS
    assert config["AD_settings_sha256"] == sha(output / "inputs/APPLICABILITY_DOMAIN_SETTINGS.json")
    models_by_target = {t: models(t) for t in SEEDS}
    ad = {}
    for item in read(output / "inputs/APPLICABILITY_DOMAIN_SETTINGS.json"):
        t = item["target"]
        assert sha(ROOT / item["path"]) == item["sha256"]
        prep = joblib.load(ROOT / item["path"])
        ad[t] = (prep, prep.transform(training(t)[FEATURES]), item["threshold_q95"])
    batch_dir = search / "prediction_batches"
    batch_dir.mkdir(exist_ok=True)
    count, start = 0, perf_counter()
    for index, chunk in enumerate(
        pd.read_csv(
            search / "CANDIDATE_INPUTS.csv",
            float_precision="round_trip",
            chunksize=config["batch_size"],
        )
    ):
        key = digest(
            {
                "configuration": config["cache_key"],
                "batch_number": index,
                "all_profiles": chunk.profile_sha256.tolist(),
            }
        )
        path = batch_dir / f"batch_{index:04d}.csv"
        meta = path.with_suffix(".json")
        reuse = False
        if meta.exists() and path.exists():
            cached = read(meta)
            if cached["cache_key"] == key:
                assert cached["sha256"] == sha(path)
                reuse = True
        if not reuse:
            columns = []
            for target in SEEDS:
                prep, z, threshold = ad[target]
                distances = pairwise_distances(prep.transform(chunk[FEATURES]), z)
                mean5 = np.sort(distances, axis=1)[:, :5].mean(axis=1)
                chunk[target + "_AD_mean5_distance"] = mean5
                chunk[target + "_AD_ratio"] = mean5 / threshold
                target_columns = []
                for seed, pipe in models_by_target[target]:
                    column = f"{target}_seed_{seed}_pg_ml"
                    chunk[column] = pipe.predict(chunk[FEATURES])
                    target_columns.append(column)
                chunk[target + "_mean_pg_ml"] = chunk[target_columns].mean(axis=1)
                chunk[target + "_sample_SD_pg_ml"] = chunk[target_columns].std(
                    axis=1, ddof=1
                )
                columns.extend(target_columns)
            chunk["all_six_nonnegative"] = chunk[columns].ge(0).all(axis=1)
            chunk["eligible"] = (
                chunk.tnf_AD_ratio.le(1)
                & chunk.il10_AD_ratio.le(1)
                & chunk.all_six_nonnegative
            )
            chunk["exclusion_reason"] = [
                ";".join(
                    reason
                    for fail, reason in [
                        (a > 1, "TNF_AD"),
                        (b > 1, "IL10_AD"),
                        (not ok, "negative_prediction"),
                    ]
                    if fail
                )
                for a, b, ok in zip(
                    chunk.tnf_AD_ratio, chunk.il10_AD_ratio, chunk.all_six_nonnegative
                )
            ]
            table(path, chunk)
            dump(
                meta,
                {
                    "cache_key": key,
                    "sha256": sha(path),
                    "candidate_n": len(chunk),
                    "model_schema_and_all_ten_inputs_in_key": True,
                },
            )
        count += len(chunk)
        print(
            f'Predicted {count:,}/{complete["candidate_n"]:,}; elapsed {perf_counter()-start:.1f} s',
            flush=True,
        )
    prediction_path = output / "results/ALL_CANDIDATES_AND_PREDICTIONS.csv"
    with prediction_path.open("wb") as destination:
        for i in range(index + 1):
            file = batch_dir / f"batch_{i:04d}.csv"
            with file.open("rb") as stream:
                if i:
                    next(stream)
                for line in stream:
                    destination.write(line)
    assert count == complete["candidate_n"]
    dump(
        output / "checks/INFERENCE_COMPLETE.json",
        {
            "predicted_n": count,
            "batch_n": index + 1,
            "prediction_file_sha256": sha(prediction_path),
            "elapsed_seconds": perf_counter() - start,
            "predictive_fits": 0,
            "preprocessing_fits": 0,
            "AD_fits": 0,
            "configuration_cache_key": config["cache_key"],
        },
    )
