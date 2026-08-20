"""Tests for the benchmark result loader and the notebook export adapter."""

import json
import shutil

import pytest

import benchmark_results as br
import export_results


# --------------------------------------------------------------------------- #
# the committed export loads
# --------------------------------------------------------------------------- #
def test_current_results_load():
    results = br.load_results()
    assert results.available, f"results failed to load: {results.problems}"
    assert not results.problems
    for name in br.REQUIRED_TABLES:
        assert not results.table(name).empty, f"{name} table is empty"


def test_manifest_identifies_the_run():
    results = br.load_results()
    manifest = results.manifest
    assert manifest["experiment_fingerprint"]
    assert manifest["run_prefix"].endswith(manifest["experiment_fingerprint"])
    assert manifest["schema_version"] == 1


def test_results_are_labelled_with_their_maturity():
    """A smoke run must not be presented as a final benchmark."""
    results = br.load_results()
    assert results.manifest["maturity"] in {"smoke_test", "full_benchmark"}
    banner = br.maturity_banner(results)
    if results.manifest["maturity"] == "smoke_test":
        assert "not intended for model ranking" in banner
        assert "n=100" in banner


def test_primary_and_secondary_metrics_stay_distinct():
    results = br.load_results()
    banner = br.maturity_banner(results)
    assert "Primary metric" in banner and "constrained" in banner
    assert "Secondary metric" in banner and "free generation" in banner
    # The accuracy table must carry constrained accuracy, not free-generation.
    columns = list(results.table("accuracy").columns)
    assert "Average Constrained Accuracy" in columns


def test_environment_reports_the_benchmark_hardware():
    results = br.load_results()
    environment = results.environment
    assert environment["gpu_name"]
    assert environment["cuda_version"]
    assert environment["pytorch_version"]
    assert environment["bitsandbytes_version"]
    markdown = br.environment_markdown(results)
    assert environment["gpu_name"] in markdown
    assert "Benchmark Environment" in markdown


# --------------------------------------------------------------------------- #
# the MedMCQA anomaly must not reappear
# --------------------------------------------------------------------------- #
def test_current_accuracy_has_no_zero_scores():
    """The corrected pipeline replaced an earlier anomalous MedMCQA 0% result."""
    accuracy = br.load_results().table("accuracy")
    numeric = accuracy.drop(columns=["Model", "Precision"]).astype(float)
    assert (numeric > 0).all().all(), f"zero-valued accuracy present:\n{accuracy}"


def test_legacy_files_are_reported_but_never_loaded(tmp_path):
    """A superseded, un-fingerprinted CSV must not become a current table."""
    source = br.DEFAULT_RESULTS_DIR
    target = tmp_path / "current"
    shutil.copytree(source, target)

    legacy = target / "clean_three_model_raw_predictions.csv"
    legacy.write_text(
        "model_configuration,benchmark,accuracy\nGemma4-E4B INT4-NF4,medmcqa,0.0\n",
        encoding="utf-8",
    )

    results = br.load_results(target)
    assert results.available
    assert "clean_three_model_raw_predictions.csv" in results.legacy_files
    # It is not exposed as a table...
    assert set(results.tables) == set(br.REQUIRED_TABLES)
    # ...and the 0.0 value never reaches the accuracy table.
    numeric = results.table("accuracy").drop(columns=["Model", "Precision"]).astype(float)
    assert (numeric > 0).all().all()
    # ...but it is disclosed as historical.
    assert "clean_three_model_raw_predictions.csv" in br.provenance_markdown(results)


def test_loose_csvs_without_a_manifest_are_refused(tmp_path):
    """Without a manifest there is no record of provenance, so nothing loads."""
    target = tmp_path / "current"
    shutil.copytree(br.DEFAULT_RESULTS_DIR, target)
    (target / "manifest.json").unlink()

    results = br.load_results(target)
    assert not results.available
    assert results.tables == {}
    assert any("manifest.json" in problem for problem in results.problems)


# --------------------------------------------------------------------------- #
# missing / broken results fail gracefully
# --------------------------------------------------------------------------- #
def test_missing_directory_fails_gracefully(tmp_path):
    results = br.load_results(tmp_path / "does_not_exist")
    assert not results.available
    assert results.problems
    assert "export_results.py" in " ".join(results.problems)
    # The presentation helpers must still work on an empty result set.
    assert br.maturity_banner(results)
    assert br.environment_markdown(results)
    assert br.problems_markdown(results)


def test_missing_declared_file_is_reported(tmp_path):
    target = tmp_path / "current"
    shutil.copytree(br.DEFAULT_RESULTS_DIR, target)
    (target / "memory_comparison.csv").unlink()

    results = br.load_results(target)
    assert "memory" not in results.tables
    assert any("memory_comparison.csv" in problem for problem in results.problems)
    # The tables that survived are still usable.
    assert not results.table("accuracy").empty
    assert results.table("memory").empty


def test_corrupt_manifest_is_reported(tmp_path):
    target = tmp_path / "current"
    shutil.copytree(br.DEFAULT_RESULTS_DIR, target)
    (target / "manifest.json").write_text("{not json", encoding="utf-8")

    results = br.load_results(target)
    assert not results.available
    assert any("could not be parsed" in problem for problem in results.problems)


# --------------------------------------------------------------------------- #
# the export adapter
# --------------------------------------------------------------------------- #
def test_export_from_notebook_is_reproducible(tmp_path):
    out_dir = tmp_path / "export"
    manifest = export_results.export_from_notebook(export_results.DEFAULT_NOTEBOOK, out_dir)

    assert manifest["source"] == "notebook_outputs"
    assert set(manifest["files"]) == set(br.REQUIRED_TABLES)

    committed = json.loads((br.DEFAULT_RESULTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_fingerprint"] == committed["experiment_fingerprint"]
    assert manifest["environment"] == committed["environment"]

    for filename in manifest["files"].values():
        fresh = (out_dir / filename).read_text(encoding="utf-8")
        stored = (br.DEFAULT_RESULTS_DIR / filename).read_text(encoding="utf-8")
        assert fresh == stored, f"{filename} drifted from the notebook outputs"


def test_export_records_no_absolute_local_path(tmp_path):
    manifest = export_results.export_from_notebook(
        export_results.DEFAULT_NOTEBOOK, tmp_path / "export"
    )
    assert manifest["source_path"] == "gemma_medical_quantization_benchmark.ipynb"


def test_export_from_missing_run_dir_raises(tmp_path):
    with pytest.raises(export_results.ExportError, match="Run folder not found"):
        export_results.export_from_run_dir(
            tmp_path / "nope", export_results.DEFAULT_NOTEBOOK, tmp_path / "out"
        )


def test_export_from_incomplete_run_dir_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "accuracy_comparison.csv").write_text("Model,Precision\n", encoding="utf-8")
    with pytest.raises(export_results.ExportError, match="missing report CSVs"):
        export_results.export_from_run_dir(
            run_dir, export_results.DEFAULT_NOTEBOOK, tmp_path / "out"
        )


def test_missing_notebook_raises(tmp_path):
    with pytest.raises(export_results.ExportError, match="Notebook not found"):
        export_results.export_from_notebook(tmp_path / "absent.ipynb", tmp_path / "out")
