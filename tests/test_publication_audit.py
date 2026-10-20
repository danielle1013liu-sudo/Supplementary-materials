"""Release checks must work without ignored local fitting/search caches."""
from src.hydrogel_immune_ml import publication_audit
from src.data import read_csv


def test_protected_manifest_does_not_require_local_work(tmp_path, monkeypatch):
    published = ['data/input.csv', 'models/model.pklz', 'outputs/shap/reference.csv',
                 'outputs/verification.json', 'environment.yml', 'requirements.txt']
    local = ['outputs/pareto/results/large.csv', 'outputs/release_build/report.json',
             'outputs/retrained_models/model.pklz', 'outputs/runtime_provenance/run.json',
             'outputs/validation/run.log']
    for name in published + local:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test fixture', encoding='utf-8')
    monkeypatch.setattr(publication_audit, 'ROOT', tmp_path)
    publication_audit.capture_hashes(tmp_path / 'checks')
    actual = set(read_csv(tmp_path / 'checks/protected_hashes_before.csv').path)
    assert actual == set(published)
