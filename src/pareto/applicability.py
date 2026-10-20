"""Revalidate the saved diagnostic transform and update Training-only distances."""

from sklearn.metrics import pairwise_distances
from .common import (CAT, DATA, FEATURES, KEY, NUM, ROOT, csv, digest, dump,
                     joblib, np, read, sha, table, training)


def prepare(output):
    """Reuse a transformer only when all learned statistics still match Training.

    Category relabels may change distances without changing the encoder's levels.
    No predictor, preprocessing estimator or diagnostic estimator is fitted.
    """
    settings = []
    for previous in read(DATA / "APPLICABILITY_DOMAIN_SETTINGS.json"):
        target = previous["target"]
        path = ROOT / f"models/{target}_DIAGNOSTIC_AD_PREPROCESSOR.joblib"
        if sha(path) != previous["sha256"]:
            raise ValueError(f"Changed diagnostic transformer: {target}")
        train = training(target)
        if train[KEY].tolist() != previous["Training_IDs"]:
            raise ValueError(f"Changed diagnostic Training membership: {target}")
        prep = joblib.load(path)
        numeric = prep.named_transformers_["numeric"]
        imputer = numeric.named_steps["median"]
        scale = numeric.named_steps["standardise"]
        raw = train[NUM].to_numpy(dtype=float)
        np.testing.assert_allclose(imputer.statistics_, np.nanmedian(raw, axis=0),
                                   rtol=0, atol=1e-12)
        filled = imputer.transform(train[NUM])
        np.testing.assert_allclose(scale.mean_, filled.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(scale.var_, filled.var(axis=0), rtol=0, atol=1e-10)
        np.testing.assert_array_equal(scale.n_samples_seen_, len(train))
        categorical = prep.named_transformers_["categorical"]
        filled_categories = categorical.named_steps["missing"].transform(train[CAT])
        for column, levels in enumerate(categorical.named_steps["OHE"].categories_):
            np.testing.assert_array_equal(levels, np.unique(filled_categories[:, column]))
        transformed = prep.transform(train[FEATURES])
        distances = pairwise_distances(transformed)
        np.fill_diagonal(distances, np.inf)
        mean5 = np.sort(distances, axis=1)[:, :5].mean(axis=1)
        threshold = float(np.quantile(mean5, .95))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"Invalid Training applicability threshold: {target}")
        table(output / "inputs" / f"{target}_TRAINING_SELF_EXCLUDED_AD.csv", {
            KEY: train[KEY], "self_excluded_mean5_distance": mean5,
            "AD_ratio": mean5 / threshold,
        })
        settings.append({
            "target": target, "path": str(path.relative_to(ROOT)),
            "sha256": sha(path), "Training_IDs": train[KEY].tolist(),
            "Training_ID_hash": digest(train[KEY].tolist()),
            "dataset_sha256": sha(ROOT / f"data/{target}.csv"),
            "transformed_Training_hash": digest(transformed),
            "threshold_q95": threshold, "neighbors": 5,
            "fit_scope": "Verified original Training-only median/scaler/OHE; current Training distances and self-excluded q95",
            "learned_statistics_revalidated": True,
            "transform_reused_without_fit": True,
            "previous_threshold_q95": previous["threshold_q95"],
        })
    dump(output / "inputs/APPLICABILITY_DOMAIN_SETTINGS.json", settings)
    return settings
