# Data dictionary and prediction scope

The approved thermal-harmonised v2 datasets contain 139 TNF-alpha and 133 IL-10 observations. `dataset_row_id` is the immutable row identifier; `corrected_value_pg_ml` is the measured response in pg/mL. Empty numeric fields remain missing. Category strings such as `none` and `unclear` remain observed levels.

| Canonical descriptor | Column | Unit/type |
| --- | --- | --- |
| Hydrogel composition | polymer_origin_class | categorical |
| Modulus | modulus_value_kPa | kPa |
| Pore size | pore_size_um | micrometres |
| Bioactive cue | material_bioactive_cue_class | categorical |
| Crosslink method | gelation_or_crosslinking_mechanism | categorical |
| Crosslink level | network_density_proxy | categorical |
| Macrophage model | macrophage_model_class | categorical |
| Activation stimulus | stimulation_context | categorical |
| Exposure mode | cell_material_interaction_mode | categorical |
| Exposure time | culture_or_exposure_duration_h | h |

Both targets use this exact feature order and the shared category codebook. Numerical preprocessing is model-specific. TNF uses median imputation; IL-10 uses the accepted IterativeImputer with BayesianRidge. Categorical preprocessing uses the fixed missing-value token and one-hot encoding. Complete constructors are in `src/models.py` and parameters in `data/model_parameters.json`.

The external-case table stores HPA@SIN-GM as Group 2 and HPA@SIN-GM@DR as Group 4. Their raw descriptor vectors match cell-by-cell across targets. RAW264_7, physical_self_assembly and the B-to-C cue contrast are preserved. Observed concentrations are 612 and 488.963 pg/mL for TNF-alpha, and 155 and 188.06 pg/mL for IL-10.

The ordered Training/Test IDs and Validation folds are in `data/splits/`. Hashes of these definitions use canonical JSON with sorted object keys and compact separators. Dataset, parameter, reference-array and fitted-model hashes use file bytes. `.gitattributes` disables line-ending conversion to preserve approved hashes when cloning.

`outputs/predictions/by_seed.csv` distinguishes Training-in-sample, Test-out-of-fit and external predictions. Training-in-sample predictions are distinct from pooled OOF Validation predictions in `outputs/validation/`. Each Test/case prediction comes from the same fitted Training-only object at that seed. Neither Test nor external-case rows are used in fitting or preprocessing estimation.

The source bibliography is in `data/references.csv`; row-level formulation, assay and extraction qualifications are in `data/record_identity.csv`. The row-based holdout and shuffled CV are not study-grouped validation.
