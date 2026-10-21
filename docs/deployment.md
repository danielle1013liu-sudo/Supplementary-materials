# Streamlit Community Cloud deployment

Live app: [Hydrogel Cytokine Explorer](https://hydrogel-cytokine-explorer.streamlit.app/).
The deployment follows the `codex/publish-hydrogel-explorer` branch of
`danielle1013liu-sudo/Supplementary-materials`.

Anonymous browser verification and downloaded-prediction comparisons are saved
under `results/cloud_deployment/`; desktop and mobile screenshots are in
`docs/cloud_deployment/browser/`. The supplement download matches the release
document bytes. Platform telemetry failures are logged separately from application
errors by the browser checks.

Deploy the website branch with **`deploy/streamlit_app.py`** as the main file and **Python 3.13** selected in Advanced settings. No API key or other secret is required for the saved-model predictions.

The entrypoint runs the existing `app.py` and explicitly selects the `community-cloud` runtime profile. It preserves the layout, theme and inference inputs. Run the same entrypoint locally from the repository root:

```sh
python -m pip install -r deploy/requirements.txt
python -m streamlit run deploy/streamlit_app.py
```

`deploy/requirements.txt` includes `../requirements-app.txt`, which includes the unchanged frozen scientific requirements and pins Streamlit. Nested requirements paths are resolved relative to the containing requirements file. A dependency file beside the entrypoint takes precedence over the root `environment.yml`; this avoids selecting the separate scientific reproduction environment. See [Streamlit dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and the [pip nested-requirements implementation](https://github.com/pypa/pip/blob/main/src/pip/_internal/req/req_file.py).

Keep `.streamlit/config.toml` at the repository root. Do not bind the hosted server to `127.0.0.1`. Community Cloud runs the entrypoint from the repository root and expects forward-slash paths. See [Streamlit file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Runtime verification

The scientific reproduction path remains unchanged: calling `load_resources()` requires Python 3.13.9 and every version in `data/software_versions.json` exactly. The website deployment profile accepts patch releases within Python 3.13, while requiring the same exact scientific package versions.

Before the cloud profile returns model resources, it verifies all existing input, model and release hashes; Training membership and applicability-domain coordinates; and predictions from every saved pipeline on the frozen Training, Test and C4 profiles. Predictions must match `data/reference/accepted_predictions.npz` exactly. A mismatch disables prediction. No training, fitting, model conversion or data edits occur.

The runtime writes its package versions to `results/runtime/environment.csv` and its prediction replay to `results/runtime/prediction_replay.csv` and `prediction_replay_checks.csv`. These files describe the actual server that executed the check. A successful local run does not substitute for checking the deployed server. The application caches validated model resources by release hashes and runtime profile.

## Required files

Publish the complete frozen release together with the website additions. The loader checks files listed in `data/checksums.json` and `data/pareto/CURRENT_RELEASE.json`, including their documentation and current snapshot dependencies; deleting unused-looking release files can prevent prediction.

The website also requires `web/`, `assets/banner.svg`, the optional `assets/banner_mobile.svg`, `docs/Supplementary_IL10_updated_20261020.docx`, `results/revision_current/numeric_display_ranges.csv`, and the application modules in `src/hydrogel_immune_ml/`. Retain live text and transparent assets when replacing the figure; the server does not regenerate artwork.

## Operations

On the deployed site, open Home and Model Prediction, submit a known profile, and download Inputs and Predictions together. Verify the downloaded predictions against the saved pipelines on those downloaded inputs. Confirm that the supplementary document downloads and that the mobile layout remains usable.

Saved-model loading and TreeSHAP consume server memory. Model resources are shared by Streamlit's resource cache, and numerical thread pools are limited during replay and prediction. Local memory measurements are deployment diagnostics, not a guarantee of Linux server memory use or concurrent-user capacity.

CSV files under `results/app/` and `results/runtime/` are generated server files. Download outputs that must be retained; do not rely on the hosted working directory as permanent storage. Inspect Cloud build/runtime logs if a dependency install, verification check or resource limit fails. Keep the scientific version pins and verification checks intact when diagnosing failures.
