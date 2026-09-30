"""Global non-dominance and deterministic representative selection."""

from .common import (
    EXPERIMENT,
    FEATURES,
    NUM,
    Path,
    SEEDS,
    csv,
    digest,
    dump,
    front,
    json,
    np,
    pd,
    read,
    sha,
    table,
)
import itertools

XY = ["tnf_mean_pg_ml", "il10_mean_pg_ml"]


def run(output):
    """Write this numerical stage below the separate output directory."""
    output = Path(output)
    OUT = output / "results"
    config = read(output / "search/SEARCH_CONFIGURATION.json")
    data = csv(OUT / "ALL_CANDIDATES_AND_PREDICTIONS.csv")
    assert (
        len(data) == config["declared_finite_grid_size"] and data.candidate_id.is_unique
    )
    eligible = data.loc[data.eligible].copy()
    universe = digest(eligible.candidate_id.tolist())
    masks = pd.DataFrame(
        {
            "candidate_id": data.candidate_id,
            "eligible": data.eligible,
            "mean_front": False,
            "nondomination_frequency_of_9": 0,
        }
    )
    masks.loc[eligible.index, "mean_front"] = front(eligible[XY[0]], eligible[XY[1]])
    pair_checks = []
    for ts, ils in itertools.product(SEEDS["tnf"], SEEDS["il10"]):
        x, y = (
            eligible[f"tnf_seed_{ts}_pg_ml"].to_numpy(),
            eligible[f"il10_seed_{ils}_pg_ml"].to_numpy(),
        )
        full_mask = front(x, y)
        union_indices = []
        for start in range(0, len(eligible), config["batch_size"]):
            stop = min(start + config["batch_size"], len(eligible))
            union_indices.extend(
                start + np.flatnonzero(front(x[start:stop], y[start:stop]))
            )
        union_indices = np.asarray(union_indices)
        merged = np.zeros(len(eligible), bool)
        merged[union_indices[front(x[union_indices], y[union_indices])]] = True
        np.testing.assert_array_equal(full_mask, merged)
        column = f"front_tnf_{ts}_il10_{ils}"
        masks[column] = False
        masks.loc[eligible.index, column] = full_mask
        masks.loc[eligible.index, "nondomination_frequency_of_9"] += full_mask.astype(
            int
        )
        pair_checks.append(
            {
                "TNF_realisation": ts,
                "IL10_realisation": ils,
                "eligible_universe_hash": universe,
                "universe_n": len(eligible),
                "front_n": int(full_mask.sum()),
                "batch_reduction_matches_full_universe": True,
            }
        )
    masks["final_front"] = masks.mean_front & masks.nondomination_frequency_of_9.ge(6)
    table(OUT / "COMPUTATIONAL_FRONT_MASKS.csv", masks)
    table(output / "checks/PAIRING_UNIVERSE_CHECKS.csv", pair_checks)
    final = data.loc[masks.final_front].copy()
    final["nondomination_frequency_of_9"] = masks.loc[
        final.index, "nondomination_frequency_of_9"
    ]
    final = final.sort_values(
        XY + ["candidate_id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    final.insert(0, "profile_number", np.arange(1, len(final) + 1))
    final["objective_coordinate_key"] = [
        float(x).hex() + "|" + float(y).hex()
        for x, y in final[XY].itertuples(index=False, name=None)
    ]
    table(OUT / "FINAL_PARETO_PROFILES.csv", final)
    table(
        OUT / "PROFILE_TO_COORDINATE_MAP.csv",
        final[["profile_number", "candidate_id", "objective_coordinate_key"] + XY],
    )
    locations = final.drop_duplicates(XY).sort_values(XY).copy()
    if len(locations) < 3:
        dump(
            OUT / "REPRESENTATIVE_LIMITATION.json",
            {
                "locations": len(locations),
                "reason": "Fewer than three distinct final locations; no representatives invented",
            },
        )
        representatives = final.iloc[:0].copy()
    else:
        first = final.sort_values(
            XY + ["candidate_id"], ascending=[True, False, True]
        ).iloc[0]
        last = final.sort_values(
            [XY[1], XY[0], "candidate_id"], ascending=[False, True, True]
        ).iloc[0]
        values = locations[XY].to_numpy()
        scaled = (values - values.min(axis=0)) / np.ptp(values, axis=0)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(scaled, axis=0), axis=1))]
        locations["normalised_cumulative_arc_length"] = arc
        locations["distance_to_half_arc_length"] = abs(arc - arc[-1] / 2)
        interior = locations.iloc[1:-1].sort_values(
            ["distance_to_half_arc_length", "candidate_id"]
        )
        middle = final.loc[final.candidate_id.eq(interior.iloc[0].candidate_id)].iloc[0]
        representatives = pd.DataFrame([first, middle, last])
        representatives.insert(0, "display_label", ["P1", "P2", "P3"])
        assert representatives.objective_coordinate_key.nunique() == 3
        table(
            OUT / "REPRESENTATIVE_ARC_LENGTH_SELECTION.csv",
            locations[
                ["candidate_id"]
                + XY
                + ["normalised_cumulative_arc_length", "distance_to_half_arc_length"]
            ],
        )
    table(OUT / "FINAL_PARETO_REPRESENTATIVES.csv", representatives)
    coverage = []
    for stage, d in [("generated", data), ("eligible", eligible), ("final", final)]:
        for field in FEATURES:
            for value, count in d[field].value_counts().sort_index().items():
                coverage.append(
                    {
                        "stage": stage,
                        "descriptor": field,
                        "value": value,
                        "unique_values": d[field].nunique(),
                        "profiles": int(count),
                    }
                )
    table(OUT / "COVERAGE_BEFORE_AND_AFTER_FILTERING.csv", coverage)
    differences = []
    if len(representatives) == 3:
        rep = representatives.set_index("display_label")
        for a, b in [("P1", "P2"), ("P2", "P3"), ("P1", "P3")]:
            for field in FEATURES:
                differences.append(
                    {
                        "comparison": b + " minus " + a,
                        "descriptor": field,
                        "first": rep.loc[a, field],
                        "second": rep.loc[b, field],
                        "changed": rep.loc[a, field] != rep.loc[b, field],
                    }
                )
    table(OUT / "REPRESENTATIVE_INPUT_DIFFERENCES.csv", differences)
    summary = {
        "generated_n": len(data),
        "predicted_n": len(data),
        "eligible_n": len(eligible),
        "excluded_n": len(data) - len(eligible),
        "negative_prediction_n": int((~data.all_six_nonnegative).sum()),
        "outside_TNF_AD_n": int(data.tnf_AD_ratio.gt(1).sum()),
        "outside_IL10_AD_n": int(data.il10_AD_ratio.gt(1).sum()),
        "mean_front_profiles": int(masks.mean_front.sum()),
        "mean_front_locations": len(data.loc[masks.mean_front, XY].drop_duplicates()),
        "final_profiles": len(final),
        "final_locations": len(locations),
        "nine_of_nine_profiles": int(final.nondomination_frequency_of_9.eq(9).sum()),
        "nine_of_nine_locations": len(
            final.loc[final.nondomination_frequency_of_9.eq(9), XY].drop_duplicates()
        ),
        "representatives": representatives[
            ["display_label", "candidate_id"] + FEATURES + XY
        ].to_dict("records"),
        "final_composition_counts": final.polymer_origin_class.value_counts().to_dict(),
        "final_context_values": {
            f: sorted(final[f].unique().tolist()) for f in EXPERIMENT + [NUM[2]]
        },
        "eligible_universe_sha256": universe,
        "prediction_source_sha256": sha(OUT / "ALL_CANDIDATES_AND_PREDICTIONS.csv"),
        "all_pairings_used_same_complete_universe": True,
        "frequency_threshold": 6,
        "model_refits": 0,
    }
    dump(OUT / "PARETO_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
