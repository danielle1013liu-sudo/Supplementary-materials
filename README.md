# Hydrogel cytokine prediction and Pareto analysis

Research code and data for predicting macrophage-associated TNF-alpha and IL-10 concentrations from ten material and experimental descriptors. The numerical workflow loads saved fitted pipelines and reproduces the completed all-ten-descriptor Pareto analysis. Plotting code and manuscript drafts are not included.

## Models and data

| Target | Model | Numerical imputation | Observations | Training / Test |
|---|---|---|---|---|
| TNF-alpha | GradientBoostingRegressor | Median | 139 | 111 / 28 |
| IL-10 | RandomForestRegressor | Iterative Bayesian ridge | 133 | 106 / 27 |

These are two model families, each with three saved fitted realisations. Complete parameters and final seeds are in `data/model_parameters.json`; row memberships and ten-fold assignments are in `data/splits/`. Models and data are checked by SHA-256 before replay.

TNF-alpha uses 1000 trees, learning rate 0.08, maximum depth 3, minimum leaf size 8 and subsample fraction 0.7. IL-10 uses 250 trees, maximum depth 9, maximum feature fraction 0.7, minimum leaf size 1 and no bootstrap sampling. Validation uses pooled out-of-fold predictions within the Training subsets. Test and case outcomes informed model selection, so the reported performance represents retrospective evaluation.

`data/tnf.csv` and `data/il10.csv` contain the corrected thermal-harmonised v2 records. Concentrations are in pg/mL, modulus in kPa, pore size in micrometres and exposure time in hours. The schema and categorical definitions are supplied in `data/feature_schema.json` and `data/category_codebook.csv`.

Some crosslink-level assignments differ between target datasets for source-matched formulations. These unresolved curation issues are described in [docs/cross_target_descriptor_notes.md](docs/cross_target_descriptor_notes.md). The release reproduces the existing numerical results without silently changing those categories; it must not be described as a fully reconciled cross-target dataset.

## Environment

Use Python 3.13.9 and the exact versions in `requirements.txt` (including scikit-learn 1.7.2 and SHAP 0.49.1). Do not upgrade libraries when replaying the saved objects. The recorded environment is Windows; other platforms must pass the numerical checks before their outputs are treated as equivalent.

```text
python -m pip install -r requirements.txt
python -m src.pareto --output outputs/all10_replay
python -m src.case_replay --output outputs/all10_replay/checks
python -m pip install -r requirements-test.txt
python -m pytest tests --replay-dir outputs/all10_replay
```

These commands perform no predictive-model fitting, preprocessing fitting, t-SNE embedding or plotting. The output directory is separate from supplied data, fitted models and reference results. Individual stages are available through `--stage generate`, `predict`, `select`, `explain` or `verify`. Prediction batches are checked against input, schema and model identities before reuse.

The pre-existing primary-model check remains available as `python -m src.verify`; it reconstructs split/fold memberships and checks saved Validation, Test and C4 results. It writes its primary-model verification under `outputs/`.

## Pareto front

Candidate generation combines Training-supported material tuples with Training-supported macrophage, stimulus, exposure-mode and exposure-time combinations. The numerical grids and category support are recorded in `data/pareto/reference/SEARCH_CONFIGURATION.json` and regenerated under the selected output directory. Every one of the ten descriptors varies before prediction.

The objectives are to minimise mean predicted TNF-alpha and maximise mean predicted IL-10. Eligibility requires both Training-reference applicability-domain ratios to be at most one and every individual prediction to be non-negative. Non-dominance is computed against the complete eligible universe. Retained profiles must belong to its mean-prediction front and be non-dominated in at least six of the nine cross-target model pairings, each using that same universe. Distinct input profiles with identical predictions remain separate.

The completed analysis generates 385,237 profiles, of which 224,895 are eligible. The final Pareto front contains 16 profiles at 14 objective locations. These counts are verification results, not selection targets.

| Label | Candidate ID | Non-dominance frequency |
|---|---|---|
| P1 | A10_1aa126e1142f7ae42d8f | 6/9 |
| P2 | A10_40b4f5f8df5f8ae9cc29 | 9/9 |
| P3 | A10_cd26a0990ac785993dcc | 9/9 |

P1 and P3 are the minimum-TNF and maximum-IL-10 endpoints. P2 is the interior objective location nearest half the cumulative arc length after min-max scaling; candidate IDs resolve design ties. P2 and P3 share modulus and pore size but differ in crosslink level and all four experimental descriptors. They do not constitute a controlled comparison of material effects or clinically optimal formulations.

Signed grouped TreeSHAP uses the saved primary trees, tree-path-dependent perturbation and raw output. One-hot contributions are summed by descriptor and then averaged across the three realisations per target. Each explanation is checked against its prediction and the saved contributions.

## Outputs and checks

`search/` contains candidate inputs, category/time support and numerical grids. `results/` contains all six prediction vectors, means, sample SDs, domain ratios, selection masks, final profiles, representatives and SHAP contributions. `checks/` contains identity, membership, additivity and numerical comparisons. Compressed reference CSVs are in `data/pareto/reference/`.

Identity sets and membership masks must match exactly. Numerical comparisons use absolute tolerance 1e-10 pg/mL (and 1e-7 for SHAP), with zero relative tolerance. The checked replay had maximum difference zero for candidate predictions, domain ratios, retained profiles and representative SHAP; the largest SHAP additivity residual was 1.23e-11 pg/mL. Focused tests also cover objective ties, global selection, all six non-negativity checks and unchanged case predictions.

C4 uses the same primary Training-only objects as Test prediction. P15 is a separate retrospective whole-study-exclusion comparison under basal THP-1 conditions at 72 h. Its saved objects are isolated in `models/p15/`; the primary Pareto analysis never loads them. All six TNF-alpha and five IL-10 observations from that study were excluded from its fitting populations, leaving 133 and 128 observations. Both P15 formulations lie outside the corresponding applicability domains.

## Optional training and grid search

```text
python -m src.train
python -m pip install -r requirements-grid.txt
python -m src.grid_search --target tnf
python -m src.grid_search --target il10
```

These are separate, optional commands. Fixed training reconstructs the accepted configuration and memberships, writing new models under `outputs/retrained_models/` without replacing distributed pipelines. Grid search uses the declared model-specific grids and final seeds; there is no seed-screening controller. The distinction between historical comparison evidence and optional reruns is documented in [docs/grid_search.md](docs/grid_search.md). Neither command was run for the present numerical replay.

## Licence and attribution

Source-study titles and DOIs are retained in `data/references.csv` and `data/record_identity.csv`. Existing third-party attributions are preserved. No repository licence is currently supplied; public visibility does not itself grant an open-source licence. No new licence is assigned here.
