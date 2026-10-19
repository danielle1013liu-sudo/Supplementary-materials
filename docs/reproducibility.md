# Numerical reproducibility

The release retains the thermal-harmonised v2 data, saved split and CV definitions, fixed preprocessing and all six accepted fitted pipelines. The interpretation modules reproduce the SHAP explanations, case studies and t-SNE coordinates used with these models. The Pareto module reproduces the all-ten-descriptor analysis in Supplementary Table S6.

## Scientific locks

| Input | SHA-256 |
| --- | --- |
| TNF-alpha dataset | 289d85de602cf9277ba2cbce368439db2019df78b58bada7200fa9e855397429 |
| IL-10 dataset | 8aef2ec5bcae3b966559531c8ecbda9ec939a68686fa300d4a1b70293d03a018 |
| External cases | 8615aca1a5935ee98ee7feeca0f8fd585beef8a55fb547ce313443eb9ccb0406 |
| Ten-feature schema | 2a285420024a61d0aed54b51e06917daca6f56527599b8b8061d6b66f95e6720 |
| Archived legacy category codebook | b10ae08d8fb5c7a83d8b9099d1cebb23851f574abfbf93aa1b409fa6c483fe00 |

`data/checksums.json` locks all scientific inputs included in this repository, including the six fitted objects. The complete accepted parameter file is unchanged. Only the manifest's scope is reduced to the files distributed in this model-only package.

The 19 October 2026 publication-code revision changes only the letters and presentation order in `data/category_codebook.csv`; its current hash is recorded in `data/checksums.json`. The original file is retained byte-for-byte as `data/public_codebooks/legacy_codes_v1.csv`, and Pareto verification checks it against the unchanged historical scientific lock. Current presentation metadata and publication exports have their own checksum entries. Definitions, boundaries, caveats, counts, canonical datasets, scientific reference outputs and fitted objects remain unchanged.

## Executable verification

`python -m src.verify` performs the following without fitting models:

1. Verify input/model hashes and pinned software versions.
2. Reconstruct the 111/28 TNF and 106/27 IL-10 memberships from split seeds 1975 and 1376, requiring the exact saved order.
3. Reconstruct shuffled ten-fold assignments from CV seeds 8 and 24, requiring exact saved membership.
4. Compare every saved estimator/preprocessor parameter with its reconstructed fixed pipeline definition.
5. Generate Test and external-case predictions from each same saved Training-only object and require exact equality with accepted arrays.
6. Check stored pooled OOF predictions and recomputed Validation/Test metrics against the accepted reference and previous fixed-training records.
7. Check the complete archived 20-competitor Validation table at each final seed and confirm the accepted pipeline is the highest-ranked model/imputer combination.
8. Calculate mean R2 and sample SD across the same three seeds per target.

Exact means elementwise equality without numerical tolerance. Prediction outputs preserve raw values. The package was checked in the pinned Windows environment; cross-platform exactness is not assumed.

## Fixed training and optional grid

The previously completed fixed-seed training reproduction produced exact Training, pooled OOF Validation, Test and case predictions for all six pipelines. Its numerical evidence is retained in `outputs/validation/reproduction.json`, the two fixed-seed reproduction CSVs and OOF tables. `python -m src.train` can repeat this check using only the accepted Training IDs and six final seeds. It writes new fitted files separately and cannot replace the distributed accepted objects.

`src/grid_search.py` is an executable refactoring of the declared model-specific local comparison. It uses fold-fitted preprocessing, identical saved folds, pooled OOF ranking and only the target's three final estimator seeds. It is not executed by default. Historical comparison results, final all-20 certification tables and previously completed controlled-grid reproduction evidence are distributed without rerunning a grid for this upload. See `grid_search.md` for the distinction between historical reference-seed evidence and optional final-seed execution.

The current code audit also reran fixed training and recovered the accepted Training, OOF, Test and C4 arrays exactly. No new grid comparison or model selection was performed. Machine-specific runtime provenance is generated locally in an ignored directory.

## Scope of inference

Validation, Test and case calculations use one accepted pipeline definition per target, with estimator seed as the only difference among its three final models. CV fits each preprocessing step within the fold; the final objects use the complete fixed 80% Training subset only. Test and case performance remains split-, Test-, and case-conditioned exploratory selection, not independent confirmation.

## Models and data

| Target | Model | Numerical imputation | Observations | Training / Test |
|---|---|---|---|---|
| TNF-alpha | GradientBoostingRegressor | Median | 139 | 111 / 28 |
| IL-10 | RandomForestRegressor | Iterative Bayesian ridge | 133 | 106 / 27 |

These are two model families, each with three saved fitted realisations. Complete parameters and final seeds are in `data/model_parameters.json`; row memberships and ten-fold assignments are in `data/splits/`. Models and data are checked by SHA-256 before replay.

TNF-alpha uses 1000 trees, learning rate 0.08, maximum depth 3, minimum leaf size 8 and subsample fraction 0.7. IL-10 uses 250 trees, maximum depth 9, maximum feature fraction 0.7, minimum leaf size 1 and no bootstrap sampling. Validation uses pooled out-of-fold predictions within the Training subsets. Test and case outcomes informed model selection, so the reported performance represents retrospective evaluation.

`data/tnf.csv` and `data/il10.csv` contain the corrected thermal-harmonised v2 records. Concentrations are in pg/mL, modulus in kPa, pore size in micrometres and exposure time in hours. The schema and categorical definitions are supplied in `data/feature_schema.json` and `data/category_codebook.csv`.

Some crosslink-level assignments differ between target datasets for source-matched formulations. These unresolved curation issues are described in [docs/cross_target_descriptor_notes.md](cross_target_descriptor_notes.md). The release reproduces the existing numerical results without silently changing those categories; it must not be described as a fully reconciled cross-target dataset.

## Publication category codes

The [supplementary document](Supplementary_frequency_ordered_20261019.docx), `data/category_codebook.csv` and [versioned mapping](../data/public_codebooks/frequency_codes_20261019.json) use `frequency_codes_20261019`. Codes are nominal labels, ordered by pooled frequency in Tables S8-S9, with other/mixed and unclear categories placed last. Counts pool all 139 TNF-alpha and 133 IL-10 target-specific records without deduplication; they are not independent studies, unique formulations or literature prevalence. Crosslink-level codes A/B/C mean medium/high/low; the original low/medium/high scoring criteria are unchanged.

[Old/new codes](../data/public_codebooks/category_code_changes_20261019.csv) retain every machine value. The original [legacy codebook](../data/public_codebooks/legacy_codes_v1.csv) preserves the historical release-lock bytes. The modelling schema, fitted encoder vocabularies and candidate-generation order remain unchanged. The compatibility machine string containing `growth_factor` retains the scientific meaning “anti-inflammatory + pro-regenerative signalling cues”.

The codebook's pre-existing development/test count columns do not consistently match the frozen release's split memberships. Both are preserved. [The partition-count audit](../results/publication_codes_20261019/category_partition_count_audit.csv) records those differences and current row IDs; aggregate historical counts do not identify earlier record-level memberships. Publication ordering uses the verified complete-target totals only.

`data/publication/*_supplementary_codes_20261019.csv` preserve the Word tables' row order, displayed numerical text, DOI cells and missing-value markers. They are publication tables. `*_model_codes_20261019.csv` preserve canonical row order, row IDs and all numeric text, with an explicit `publication_codebook_version` column. These model-compatible exports decode back to canonical inputs before frozen-model prediction. The shared adapter preserves valid `none`, true missing values and both historical categorical-missing tokens separately.

```text
python -m src.public_predict data/tnf.csv --target tnf --output results/tnf_predictions.csv
python -m src.public_predict data/publication/tnf_model_codes_20261019.csv --target tnf --codebook-version frequency_codes_20261019 --output results/tnf_coded_predictions.csv
python -m src.hydrogel_immune_ml.publication_exports
python -m src.hydrogel_immune_ml.supplementary
```

Canonical machine values remain the default. Letter-coded imports require the matching explicit version; `legacy_codes_v1` supports original letters. Invalid codes, conflicting version metadata and attempts to encode an already coded table fail. The document migration identifies the complete verified document version and leaves an already revised document unchanged. The original source is retained in `docs/source/`. The supplied frequency-ordered document and dated CSV package were subsequently located and checked against these outputs: all table cells, category mappings, counts and embedded figures agree. The document packages differ in XML serialization and in the hyphen used in the coding note; the scientific contents are unchanged.

Publication verification results are in [results/publication_codes_20261019](../results/publication_codes_20261019). Run `python -m src.hydrogel_immune_ml.publication_audit` for frozen-model round trips, isolated original-release verification, separate P15 replay and checks of stored Pareto/SHAP tables. This performs no fitting or candidate generation. Then run `python -m pytest tests --replay-dir tmp/publication_codes_20261019/pareto`. Existing numerical references and all protected scientific files retain their original bytes.

The pinned-environment check passed all 35 tests without skips. All 82 protected scientific files were byte-identical. Maximum differences were zero for all six models' round-trip transforms and predictions, CLI imports, stored OOF/Test/C4 and separate P15 predictions, saved-domain round trips, and stored Pareto/SHAP comparisons. The unchanged stored SHAP additivity residual was 1.2278178473934531e-11 pg/mL, within the existing 1e-7 tolerance. All 1,904 categorical cells were checked, and every page of the 21-page Word render was visually inspected against the original. No new fits, candidate generation or SHAP analysis were performed.

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

The current figures and publication-coded profile tables are in [`results/pareto`](../results/pareto). To render them from the supplied numerical references:

```sh
python -m pip install -r requirements-plots.txt
python -m src.pareto.plotting --output results/pareto
```

After a complete numerical replay, add `--input outputs/all10_replay/results` to plot its results instead. Both paths use the same profile identities and representative selection.

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

## SHAP, case studies and t-SNE

Run these commands from the repository root after installing `requirements.txt`:

```sh
python -m src.shap_analysis
python -m src.case_studies
python -m src.tsne_analysis
python -m pip install -r requirements-plots.txt
python -m src.plot_interpretation
```

The numerical outputs are in [`outputs/shap`](../outputs/shap), [`outputs/case_studies`](../outputs/case_studies) and [`outputs/tsne`](../outputs/tsne). Figures are in [`outputs/figures`](../outputs/figures). Each numerical command accepts `--output`; t-SNE also accepts `--shap-dir`. The plotting command accepts `--input` for an alternative outputs root and `--output` for the figure directory.

`shap_analysis` loads the six primary fitted objects and exports transformed-column and ten-descriptor contributions. It checks their exact equality with the archived SHAP arrays and verifies additivity. Global importance uses Test observations only; the all-row explanations are descriptive and include in-sample Training observations. They are not out-of-fold explanations.

`case_studies` reproduces the four formulations in Supplementary Table S7. P15 uses separate whole-study-excluded models; C4 uses the primary Training-only objects. Predictions, individual-model variation, local SHAP and paired SHAP changes are exported separately. Both P15 formulations fall outside their corresponding applicability domains. Observed C4 intervals retain the source-figure estimates and archived asymmetric ranges; they should not all be interpreted as experimental standard deviations.

`tsne_analysis` embeds the unstandardised, signed consensus SHAP vectors, with separate Test-only and all-row scopes. All twelve embeddings are checked against the archived numerical coordinates. The all-row plots use perplexity 30; ten-neighbour trustworthiness is 0.923677 for TNF-alpha and 0.977130 for IL-10, matching the rounded values in Supplementary Figures S2-S3. Five-neighbour diagnostics are reported separately. No external cases are embedded.

The t-SNE step fits an embedding, but none of these commands refits predictive models or their preprocessing. Figure legends use the current publication codebook. Reference provenance and hashes are recorded in [`data/interpretation`](../data/interpretation).

## Outputs and checks

`search/` contains candidate inputs, category/time support and numerical grids. `results/` contains all six prediction vectors, means, sample SDs, domain ratios, selection masks, final profiles, representatives and SHAP contributions. `checks/` contains identity, membership, additivity and numerical comparisons. Compressed reference CSVs are in `data/pareto/reference/`.

Identity sets and membership masks must match exactly. Numerical comparisons use absolute tolerance 1e-10 pg/mL (and 1e-7 for SHAP), with zero relative tolerance. The checked replay had maximum difference zero for candidate predictions, domain ratios, retained profiles and representative SHAP; the largest SHAP additivity residual was 1.23e-11 pg/mL. Focused tests also cover objective ties, global selection, all six non-negativity checks and unchanged case predictions.

C4 uses the same primary Training-only objects as Test prediction. P15 is a separate retrospective whole-study-exclusion comparison under basal THP-1 conditions at 72 h. Its saved objects are isolated in `models/p15/`; the primary Pareto analysis never loads them. All six TNF-alpha and five IL-10 observations from that study were excluded from its fitting populations, leaving 133 and 128 observations. Both P15 formulations lie outside the corresponding applicability domains.
