"""Tests that the model registry matches the identifiers used by the project."""

import json

import pytest

import benchmark_results as br
import model_registry as registry
import prompts

NOTEBOOK_BASE_IDS = {
    "Gemma4-E4B": "google/gemma-4-E4B-it",
    "Gemma4-12B": "google/gemma-4-12B-it",
    "MedGemma": "google/medgemma-1.5-4b-it",
}

# Repository IDs recorded by the notebook upload cell.
NOTEBOOK_QUANTIZED_IDS = {
    ("Gemma4-E4B", "INT8"): "pupupapapa/gemma-4-e4b-it-int8-bnb",
    ("Gemma4-E4B", "INT4-NF4"): "pupupapapa/gemma-4-e4b-it-int4-bnb",
    ("Gemma4-12B", "INT8"): "pupupapapa/gemma-4-12b-it-int8-bnb",
    ("Gemma4-12B", "INT4-NF4"): "pupupapapa/gemma-4-12b-it-int4-bnb",
    ("MedGemma", "INT8"): "pupupapapa/medgemma-1.5-4b-it-int8-bnb",
    ("MedGemma", "INT4-NF4"): "pupupapapa/medgemma-1.5-4b-it-int4-bnb",
}


def test_registry_covers_every_benchmarked_configuration():
    assert len(registry.MODELS) == 9
    keys = {(entry.family, entry.precision) for entry in registry.MODELS}
    expected = {
        (family, precision)
        for family in NOTEBOOK_BASE_IDS
        for precision in ("Baseline", "INT8", "INT4-NF4")
    }
    assert keys == expected


def test_base_model_ids_match_the_notebook():
    for entry in registry.MODELS:
        assert entry.base_model_id == NOTEBOOK_BASE_IDS[entry.family]


def test_baseline_entries_point_at_the_upstream_repository():
    for entry in registry.MODELS:
        if entry.precision != "Baseline":
            continue
        assert entry.repo_id == NOTEBOOK_BASE_IDS[entry.family]
        assert not entry.is_private
        assert not entry.loads_prequantized


def test_quantized_entries_use_the_uploaded_repository_ids():
    for entry in registry.MODELS:
        if entry.precision == "Baseline":
            continue
        expected = NOTEBOOK_QUANTIZED_IDS[(entry.family, entry.precision)]
        assert entry.repo_id == expected, f"{entry.label} points at {entry.repo_id}"
        assert entry.is_private, "the notebook pushed these with HF_PRIVATE = True"
        assert entry.requires_cuda, "BitsAndBytes checkpoints need CUDA"
        assert entry.loads_prequantized


def test_no_invented_namespaces():
    allowed_namespaces = {"google", registry.QUANTIZED_NAMESPACE}
    for entry in registry.MODELS:
        namespace = entry.repo_id.split("/")[0]
        assert namespace in allowed_namespaces, f"unexpected namespace in {entry.repo_id}"


# --------------------------------------------------------------------------- #
# registry <-> benchmark table cross-reference
# --------------------------------------------------------------------------- #
def test_every_model_maps_to_a_benchmark_row():
    """Labels must resolve to a row in the exported benchmark tables."""
    results = br.load_results()
    assert results.available

    for entry in registry.MODELS:
        row = br.benchmark_row_for(results, entry.benchmark_row, entry.benchmark_precision)
        assert row, f"no benchmark row for {entry.label}"
        assert "Average Constrained Accuracy" in row
        assert "Peak VRAM" in row


def test_report_labels_match_the_manifest_families():
    manifest = json.loads(
        (br.DEFAULT_RESULTS_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["model_families"]) == set(registry.FAMILY_REPORT_LABELS)
    for family, config in manifest["model_families"].items():
        assert config["model_id"] == NOTEBOOK_BASE_IDS[family]
    assert set(manifest["precisions"]) == {"Baseline", "INT8", "INT4-NF4"}


# --------------------------------------------------------------------------- #
# lookup behaviour
# --------------------------------------------------------------------------- #
def test_lookup_by_key_and_by_label():
    entry = registry.default_model()
    assert registry.get_model(entry.key) is entry
    assert registry.get_model(entry.label) is entry


def test_unknown_selection_raises():
    with pytest.raises(KeyError):
        registry.get_model("Not A Model")
    with pytest.raises(KeyError):
        registry.get_model("")


def test_default_is_an_int4_configuration():
    """The benchmark shows INT4 as the cheapest configuration to serve."""
    entry = registry.default_model()
    assert entry.precision == "INT4-NF4"
    assert entry.label in registry.choice_labels()


def test_default_model_is_the_lowest_vram_configuration():
    results = br.load_results()
    memory = results.table("memory")
    lowest = memory.loc[memory["Peak VRAM"].astype(float).idxmin()]
    entry = registry.default_model()
    assert entry.benchmark_row == lowest["Model"]
    assert entry.benchmark_precision == lowest["Precision"]


def test_choice_labels_are_unique_and_list_quantized_first():
    labels = registry.choice_labels()
    assert len(labels) == len(set(labels)) == len(registry.MODELS)
    baselines = [label for label in labels if "Baseline" in label]
    first_baseline = labels.index(baselines[0])
    assert all("Baseline" in label for label in labels[first_baseline:])


# --------------------------------------------------------------------------- #
# use cases
# --------------------------------------------------------------------------- #
def test_all_four_use_cases_are_available():
    keys = {use_case.key for use_case in prompts.USE_CASES}
    assert keys == {"general_qa", "explain_term", "disease_info", "nutrition"}


def test_use_case_lookup_and_prompt_assembly():
    use_case = prompts.get_use_case("Disease Information")
    assert use_case.key == "disease_info"
    system_prompt = prompts.build_system_prompt(use_case)
    assert "Common risk factors" in system_prompt
    assert "Do not tell the user what condition they have" in system_prompt


def test_unknown_use_case_raises():
    with pytest.raises(KeyError):
        prompts.get_use_case("Prescribe Medication")


def test_messages_are_single_turn():
    """History must not be folded into the request."""
    use_case = prompts.get_use_case("general_qa")
    messages = prompts.build_messages("What is hypertension?", use_case)
    assert len(messages) == 2
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "What is hypertension?"
