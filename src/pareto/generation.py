"""Training-supported candidate enumeration."""

from .common import (
    CAT,
    DATA,
    EXPERIMENT,
    FEATURES,
    KEY,
    MATERIAL,
    NUM,
    Path,
    ROOT,
    SCHEMA,
    SEEDS,
    csv,
    dataset,
    digest,
    dump,
    joblib,
    json,
    models,
    np,
    pd,
    read,
    sha,
    table,
    training,
)
import itertools
from ..data import check_inputs

SCHEMA_VERSION = "thermal_harmonisation_v2_all10_search_v1"


def verify_models(output):
    """Verify the accepted release, including exact held-out predictions."""
    from ..data import check_environment, configuration, split_definition
    from .applicability import prepare

    check_environment()
    check_inputs()
    records = read(ROOT / "models/manifest.json")
    accepted = np.load(ROOT / "data/reference/accepted_predictions.npz")
    checks, learned = [], {}
    for target in SEEDS:
        d = dataset(target)
        ctx = split_definition(target)
        if not set(ctx["Training_IDs"]).isdisjoint(ctx["Test_IDs"]):
            raise ValueError("Training and Test memberships overlap")
        assert digest(ctx["Training_IDs"]) == ctx["Training_ID_hash"]
        assert digest(ctx["Validation_fold"]) == ctx["Validation_fold_assignment_hash"]
        assert set(ctx["Validation_fold"]) == set(range(1, 11))
        for seed, pipe in models(target):
            item = next(m for m in records if m["target"] == target and m["seed"] == seed)
            path = ROOT / item["path"]
            assert sha(path) == item["sha256"]
            pp = pipe.named_steps["preprocess"]
            assert list(pp.feature_names_in_) == FEATURES
            test = d.set_index(KEY).loc[ctx["Test_IDs"], FEATURES]
            np.testing.assert_array_equal(pipe.predict(test), accepted[f"{target}_{seed}_Test"])
            encoder = pp.named_transformers_["categorical"].named_steps["onehot"]
            columns = next(cols for name, _, cols in pp.transformers_ if name == "categorical")
            levels = dict(zip(columns, [set(a) for a in encoder.categories_]))
            if target in learned:
                assert levels == learned[target]
            learned[target] = levels
            checks.append({
                "target": target, "realisation": seed, "model_sha256": sha(path),
                "preprocessor_sha256": digest({"canonical_joblib_sha1": joblib.hash(pp, hash_name="sha1"),
                                                "contract": "preprocessor learned-state v1"}),
                "parameters_match": True, "Test_predictions_exact": True,
                "Training_n": len(ctx["Training_IDs"]), "family": configuration(target)["family"],
            })
    ad_checks = prepare(output)
    report = {
        "models": checks, "software_versions": read(ROOT / "data/software_versions.json"),
        "applicability_domain": ad_checks, "accepted_configuration": configuration(),
        "source_sha256": {name: sha(ROOT / name) for name in (
            "data/tnf.csv", "data/il10.csv", "data/model_parameters.json", "models/manifest.json",
            "data/feature_schema.json", "data/splits/tnf.json", "data/splits/il10.json",
            "data/reference/accepted_predictions.npz")},
        "predictive_fits": 0, "diagnostic_fits": 0,
    }
    dump(output / "checks/MODEL_AND_DATA_VERIFICATION.json", report)
    return learned, checks


def supported_tuples(training_sets, fields, allowed, prefix):
    observed = sorted(
        set(
            tuple(row)
            for d in training_sets.values()
            for row in d[fields].dropna().itertuples(index=False, name=None)
        )
    )
    retained, evidence = [], []
    for values in observed:
        item = dict(zip(fields, values))
        excluded = [f + "=" + str(v) for f, v in item.items() if v not in allowed[f]]
        for target, d in training_sets.items():
            matched = np.logical_and.reduce(
                [d[f].eq(item[f]).to_numpy() for f in fields]
            )
            item[target + "_Training_n"] = int(matched.sum())
            item[target + "_Training_IDs"] = json.dumps(d.loc[matched, KEY].tolist())
        item["included"] = not excluded
        item["exclusion_reason"] = "; ".join(excluded)
        item["tuple_id"] = prefix + digest(values)[:12]
        evidence.append(item)
        if not excluded:
            retained.append(item)
    return retained, evidence


def run(output):
    """Write this numerical stage below the separate output directory."""
    output = Path(output)
    OUT = output / "search"
    learned, checks = verify_models(output)
    training_sets = {t: training(t) for t in SEEDS}
    allowed, level_records = {}, []
    for field in CAT:
        allowed[field] = set()
        for value in SCHEMA["allowed_categories"][field]:
            counts = {
                t: int(d[field].eq(value).sum()) for t, d in training_sets.items()
            }
            known = all(value in learned[t][field] for t in SEEDS)
            reasons = []
            if value.lower() in [
                "unclear",
                "missing",
                "unreported",
                "unknown",
                "__missing__",
                "**missing**",
            ]:
                reasons.append(
                    "Unresolved condition is not an optimisable intervention"
                )
            if not all(counts.values()):
                reasons.append(
                    "No Training observations for "
                    + ", ".join(t for t, n in counts.items() if n == 0)
                )
            if not known:
                reasons.append("Not recognised by both fitted encoders")
            if not reasons:
                allowed[field].add(value)
            level_records.append(
                {
                    "descriptor": field,
                    "value": value,
                    **{t + "_Training_n": n for t, n in counts.items()},
                    "both_encoders_recognise": known,
                    "included": not reasons,
                    "reason": "; ".join(reasons),
                }
            )
    table(OUT / "CATEGORICAL_LEVEL_SUPPORT.csv", level_records)
    materials, material_evidence = supported_tuples(
        training_sets, MATERIAL, allowed, "M"
    )
    experiments, experiment_evidence = supported_tuples(
        training_sets, EXPERIMENT, allowed, "E"
    )
    table(OUT / "MATERIAL_TUPLE_SUPPORT.csv", material_evidence)
    table(OUT / "EXPERIMENTAL_TUPLE_SUPPORT.csv", experiment_evidence)
    grids = {}
    old_grid = csv(DATA / "EMPIRICAL_QUANTILE_GRID_20_POSITIONS.csv")
    grid_checks = []
    for field in NUM[:2]:
        lower = max(d[field].dropna().quantile(0.05) for d in training_sets.values())
        upper = min(d[field].dropna().quantile(0.95) for d in training_sets.values())
        pool = np.concatenate(
            [d[field].dropna().to_numpy() for d in training_sets.values()]
        )
        pool = pool[(pool >= lower) & (pool <= upper)]
        calculated = np.quantile(pool, np.linspace(0, 1, 20))
        calculated[0], calculated[-1] = lower, upper
        recorded = old_grid.loc[old_grid.descriptor.eq(field), "grid_value"].to_numpy()
        np.testing.assert_allclose(calculated, recorded, rtol=0, atol=1e-11)
        grids[field] = np.unique(recorded)
        grid_checks.append(
            {
                "descriptor": field,
                "lower": lower,
                "upper": upper,
                "values": grids[field],
                "max_reconstruction_residual": float(
                    np.max(abs(calculated - recorded))
                ),
            }
        )
    timefield = NUM[2]
    common_times = set.intersection(
        *(set(d[timefield].dropna().astype(float)) for d in training_sets.values())
    )
    time_evidence, contexts = [], []
    for experiment in experiments:
        associated = set()
        masks = {}
        for target, d in training_sets.items():
            masks[target] = np.logical_and.reduce(
                [d[f].eq(experiment[f]) for f in EXPERIMENT]
            )
            associated.update(d.loc[masks[target], timefield].dropna().astype(float))
        for value in sorted(associated):
            item = {f: experiment[f] for f in EXPERIMENT}
            item.update(
                {timefield: value, "experimental_tuple_id": experiment["tuple_id"]}
            )
            for target, d in training_sets.items():
                matched = masks[target] & d[timefield].eq(value)
                item[target + "_Training_n"] = int(matched.sum())
                item[target + "_Training_IDs"] = json.dumps(
                    d.loc[matched, KEY].tolist()
                )
            item["included"] = value in common_times
            item["reason"] = (
                ""
                if item["included"]
                else "Exposure time lacks observed Training support in both target domains"
            )
            time_evidence.append(item)
            if item["included"]:
                contexts.append(item)
    table(OUT / "EXPERIMENT_TIME_SUPPORT.csv", time_evidence)
    config = {
        "schema_version": SCHEMA_VERSION,
        "ordered_features": FEATURES,
        "dataset_sha256": {t: sha(ROOT / f"data/{t}.csv") for t in SEEDS},
        "schema_sha256": sha(ROOT / "data/feature_schema.json"),
        "accepted_configuration_sha256": sha(ROOT / "data/model_parameters.json"),
        "AD_settings_sha256": sha(output / "inputs/APPLICABILITY_DOMAIN_SETTINGS.json"),
        "accepted_seeds": SEEDS,
        "model_sha256": {
            f'{c["target"]}_{c["realisation"]}': c["model_sha256"] for c in checks
        },
        "numeric_grids": grid_checks,
        "common_observed_exposure_times_h": sorted(common_times),
        "n_material_tuples": len(materials),
        "n_experimental_tuples": len(experiments),
        "n_experiment_time_combinations": len(contexts),
        "rules": [
            "Material tuples are the union observed in either original Training subset, retaining levels supported in both targets and both saved encoders.",
            "Experimental triplets are jointly observed in either Training subset. Unclear or missing categories are excluded; none and basal remain.",
            "Time must be observed in both target Training domains and paired with that experimental triplet in at least one Training subset.",
            "Every retained material tuple is crossed with every retained experimental-time tuple and both verified numerical grids. No further material-context exclusion has a documented basis in the codebook; none is invented.",
            "Observed experimental triplets preserve supported cell-model/exposure compatibility. Residual other categories are retained as coded, without treating them as specific biochemical interventions.",
            "No Test or case outcomes are used to define the search. No minimum category count or composition quota is applied.",
        ],
        "representative_rule": {
            "P1": "minimum mean TNF, then maximum mean IL10, then candidate ID",
            "P3": "maximum mean IL10, then minimum mean TNF, then candidate ID",
            "P2": "interior location nearest half cumulative arc length after objective min-max scaling and ascending TNF ordering; endpoints excluded; remaining ties by candidate ID",
        },
        "dominance": "minimise TNF and maximise IL10; no worse on both, strict improvement on at least one; exact ties retained",
        "eligibility": "both saved Training-reference AD ratios <=1 and all six predictions >=0",
        "final_membership": "mean-front membership across all eligible candidates AND non-dominance in at least 6 of 9 model pairings against that same universe",
        "batch_size": 16384,
        "prediction_fits": 0,
    }
    n = len(materials) * len(contexts) * len(grids[NUM[0]]) * len(grids[NUM[1]])
    config["declared_finite_grid_size"] = n
    config["cache_key"] = digest(config)
    dump(OUT / "SEARCH_CONFIGURATION.json", config)
    print(
        f"Declared complete grid: {len(materials)} material tuples x {len(contexts)} experimental-time tuples x {len(grids[NUM[0]])} modulus x {len(grids[NUM[1]])} pore-size values = {n:,} candidates",
        flush=True,
    )
    rows = []
    for material, context, m, p in itertools.product(
        materials, contexts, grids[NUM[0]], grids[NUM[1]]
    ):
        values = {
            **{f: material[f] for f in MATERIAL},
            **{f: context[f] for f in EXPERIMENT},
            NUM[0]: float(m),
            NUM[1]: float(p),
            NUM[2]: float(context[NUM[2]]),
        }
        ordered = [values[f] for f in FEATURES]
        identity = digest({"schema_version": SCHEMA_VERSION, "profile": ordered})
        rows.append(
            {
                "candidate_id": "A10_" + identity[:20],
                **values,
                "material_tuple_id": material["tuple_id"],
                "experimental_tuple_id": context["experimental_tuple_id"],
                "profile_sha256": identity,
            }
        )
    candidates = pd.DataFrame(rows)[
        ["candidate_id"]
        + FEATURES
        + ["material_tuple_id", "experimental_tuple_id", "profile_sha256"]
    ]
    assert (
        len(candidates) == n
        and candidates.candidate_id.is_unique
        and not candidates[FEATURES].duplicated().any()
    )
    coverage = []
    for f in FEATURES:
        for value, count in (
            candidates[f].value_counts(dropna=False).sort_index().items()
        ):
            coverage.append(
                {
                    "descriptor": f,
                    "value": value,
                    "unique_values": candidates[f].nunique(),
                    "minimum": candidates[f].min() if f in NUM else "",
                    "maximum": candidates[f].max() if f in NUM else "",
                    "candidate_count": int(count),
                    "exclusions_record": "CATEGORICAL_LEVEL_SUPPORT.csv; MATERIAL_TUPLE_SUPPORT.csv; EXPERIMENTAL_TUPLE_SUPPORT.csv; EXPERIMENT_TIME_SUPPORT.csv",
                }
            )
        assert candidates[f].nunique() > 1, (f, "Search domain is constant")
    table(OUT / "SEARCH_SPACE_COVERAGE.csv", coverage)
    table(OUT / "CANDIDATE_INPUTS.csv", candidates)
    dump(
        OUT / "GENERATION_COMPLETE.json",
        {
            "candidate_n": n,
            "candidate_table_sha256": sha(OUT / "CANDIDATE_INPUTS.csv"),
            "configuration_sha256": sha(OUT / "SEARCH_CONFIGURATION.json"),
            "coverage_sha256": sha(OUT / "SEARCH_SPACE_COVERAGE.csv"),
            "all_ten_descriptors_vary": True,
            "inference_not_started": True,
            "unique_values": candidates[FEATURES].nunique().to_dict(),
            "context_values": {
                f: sorted(candidates[f].unique().tolist())
                for f in EXPERIMENT + [NUM[2]]
            },
            "source_supported_material_tuples_before_exclusions": len(
                material_evidence
            ),
            "source_supported_experimental_tuples_before_exclusions": len(
                experiment_evidence
            ),
            "excluded_material_tuples": len(material_evidence) - len(materials),
            "excluded_experimental_tuples": len(experiment_evidence) - len(experiments),
            "excluded_experiment_time_tuples": sum(
                not r["included"] for r in time_evidence
            ),
        },
    )
    print(json.dumps(read(OUT / "GENERATION_COMPLETE.json"), indent=2), flush=True)
