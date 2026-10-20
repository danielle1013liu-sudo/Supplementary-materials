# Reproducing the released models

The current IL-10 release is `IL10-crosslink7-RF-median-20261020`: a random forest with median numerical imputation, 250 trees, depth 8, `max_features=0.7` and no bootstrap sampling. TNF retains its earlier gradient boosting models. Exact parameters and realisation identifiers are in `data/model_parameters.json`; package versions are pinned in the requirements and software-version files.

IL-10 changes only seven `network_density_proxy` cells: IL10-0003-0005 become medium, IL10-0016-0018 become high, and IL10-0078 becomes low. Responses, row order, the 106/27 Training/Test split and the ten Validation folds are unchanged. Six changed rows are in Training; IL10-0017 is in Test. The change ledger, input hash, source certificates and exact replay checks are in `data/validation_evidence/il10_20261020/`.

Run from the repository root in the pinned environment:

```bash
python -m src.predict
python -m src.verify
python -m src.train --target il10
python -m src.hydrogel_immune_ml.model_comparison
```

`predict` and `verify` use the saved models without fitting. `train` refits the accepted pipeline using only saved Training rows, refits preprocessing within each Validation fold, and requires exact equality to frozen Training, pooled OOF, Test and C4 arrays. It writes reproduced models under `outputs/retrained_models/`. Input CSVs use the same pandas parsing settings used during fitting; prediction exports are read with round-trip precision.

For IL-10, family and imputer are selected together by the highest mean pooled OOF R-squared across the three shared realisations and all 20 locked configurations. Mean RMSE, mean MAE and configuration identity resolve ties. The accepted pipeline need not win for every individual realisation. Each family benchmark uses its own best imputer under the same mean-Validation rule. TNF retains its original requirement that its accepted pipeline win at each realisation.

The three IL-10 realisations were accepted after C4 case-error ranking. C4 therefore informed selection and is not an untouched external validation set. Test predictions were also computed during the seed screen, although Test was not the final family/imputer selection criterion. Report these selection conditions when interpreting the displayed performance.

The current all-20 comparison and four-family benchmark files are `final_seed_all20_comparisons.csv` and `final_family_benchmarks.csv`. Earlier `historical_local_grid.csv` and `controlled_grid_*` files are historical evidence from before the seven corrections; their IL-10 rows are not the current release. The revised reference grid and certificates are preserved separately under `il10_20261020/`.


Install the pinned runtime and optional plotting/document dependencies with:

```bash
python -m pip install -r requirements-test.txt -r requirements-plots.txt
```

To reproduce the current explanations, embeddings, figures and full candidate search:

```bash
python -m src.shap_analysis
python -m src.tsne_analysis
python -m src.case_studies
python -m src.plot_interpretation
python -m src.pareto --output outputs/pareto --reference data/pareto/current/results
python -m src.pareto.plotting --input outputs/pareto/results --output results/pareto
python -m src.hydrogel_immune_ml.supplementary --verify
python -m pytest tests --replay-dir outputs/pareto
```

The SHAP and case commands check predictions and additive contributions against versioned references. The t-SNE command checks all twelve prescribed embeddings. Global importance percentages are the normalized mean absolute signed-consensus SHAP values over Test observations, summing to 100% for each target. The all-row landscapes are descriptive and include in-sample Training explanations.

P15 excludes every available record from the source study before fitting its separate sensitivity models (six TNF and five IL-10 records). Both P15 formulations remain outside the corresponding applicability domains. With the revised IL-10 model their absolute errors increased; the smaller C4 errors do not establish a general improvement across external formulations.

The explicit release builder `src.rebuild_il10_interpretation` refits the three IL-10 P15 models and replaces IL-10 interpretation reference files after verifying the accepted primary-model locks. It is a maintainer operation, separate from ordinary replay. The normal SHAP, case, and Pareto commands use saved fitted objects.

The delivered Word document is verified in this replay. Regenerating it is a release-building operation described in `supplementary_build.txt`; refresh its recorded release checksums after an intentional document change. The publication audit (`python -m src.hydrogel_immune_ml.publication_audit`) additionally checks saved artifact bytes and requires no ignored local search cache.
