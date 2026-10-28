"""The user guide preserves category meaning without publishing workflow metadata."""

import json

from hydrogel.codebook import codebook_json, descriptor_codebook
from hydrogel.data import load_schema


def test_codebook_matches_feature_order_units_and_categories():
    schema = load_schema()
    guide = descriptor_codebook()
    assert [row["column"] for row in guide["descriptors"]] == schema["predictors"]
    expected_units = {"Gprime_kPa": "kPa", "pore_size_um": "µm", "exposure_time_h": "h"}
    for row in guide["descriptors"]:
        assert row["name"] and row["definition"]
        if row["column"] in schema["numeric"]:
            assert row["unit"] == expected_units[row["column"]]
            assert row["type"] == "numerical"
            assert "categories" not in row
        else:
            assert row["type"] == "categorical"
            assert {item["code"]: item["label"] for item in row["categories"]} == schema["categorical"][row["column"]]["labels"]


def test_codebook_uses_only_public_fields():
    guide = json.loads(codebook_json())
    assert set(guide) == {"title", "descriptors", "missing_values"}
    for row in guide["descriptors"]:
        assert set(row) <= {"column", "name", "type", "unit", "definition", "categories"}
    text = json.dumps(guide).lower()
    for forbidden in ["models_trained", "future fitting", "approval", "identity_review_status",
                      "data_revision", "source_freeze_status", "preprocessing_fit_scope"]:
        assert forbidden not in text
    assert "empty" in guide["missing_values"]["numerical"]
    assert "do not substitute zero" in guide["missing_values"]["numerical"]
    assert "unknown" in guide["missing_values"]["categorical"]


def test_user_guide_does_not_mutate_frozen_schema():
    before = load_schema()
    descriptor_codebook()
    assert load_schema() == before
