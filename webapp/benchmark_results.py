"""Load exported benchmark results for the Benchmark Explorer tab.

Results are read from ``webapp/results/current`` and are produced by
``export_results.py``. No benchmark number is defined in this module or in
``app.py``.

Two rules matter here:

* A ``manifest.json`` is required. Loose CSV files with no manifest are never
  loaded, because there is then no record of which run they came from.
* Files whose names carry no experiment fingerprint belong to superseded runs.
  They are reported as historical and never merged into the current tables. The
  known example is the earlier ``clean_three_model_*`` output, which contained an
  anomalous MedMCQA result that the corrected pipeline replaced.

The loader never raises. Problems are collected and shown in the UI so a missing
or stale export is visible rather than silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

WEBAPP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = WEBAPP_DIR / "results" / "current"

REQUIRED_TABLES = ("accuracy", "performance", "memory", "tradeoff")

TABLE_TITLES = {
    "accuracy": "Accuracy",
    "performance": "Performance",
    "memory": "Memory",
    "tradeoff": "Quantization Trade-off",
}

# Superseded artifact names: the corrected pipeline writes
# ``<prefix>_<fingerprint>_*``, so an un-fingerprinted name is an older run.
LEGACY_FILENAMES = frozenset(
    {
        "clean_three_model_raw_predictions.csv",
        "clean_three_model_benchmark_summary.csv",
        "clean_three_model_performance_summary.csv",
        "clean_three_model_memory_summary.csv",
        "clean_three_model_summary.csv",
    }
)

LEGACY_NOTE = (
    "These files come from an earlier, superseded run. They are listed for "
    "provenance only and are not shown as current results. The earlier run "
    "included an anomalous MedMCQA score that the corrected pipeline replaced."
)


@dataclass
class BenchmarkResults:
    """Everything the Benchmark Explorer needs, plus any problems found."""

    results_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    legacy_files: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.tables)

    @property
    def environment(self) -> dict[str, Any]:
        return self.manifest.get("environment") or {}

    def table(self, name: str) -> pd.DataFrame:
        """Return one table, or an empty frame when it is missing."""
        return self.tables.get(name, pd.DataFrame())


def _read_csv(path: Path, problems: list[str]) -> pd.DataFrame | None:
    try:
        # Read as text so the notebook's own formatting survives: pandas would
        # otherwise turn "0.3300" into 0.33 and "0.0000" into 0, which makes the
        # displayed table disagree with the notebook it came from.
        frame = pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        problems.append(f"Could not read `{path.name}`: {type(exc).__name__}: {exc}")
        return None
    if frame.empty:
        problems.append(f"`{path.name}` contains no rows.")
        return None
    return frame


def load_results(results_dir: Path | str | None = None) -> BenchmarkResults:
    """Load the exported benchmark results.

    Args:
        results_dir: Directory holding ``manifest.json`` and the report CSVs.
            Defaults to ``webapp/results/current``.

    Returns:
        A :class:`BenchmarkResults`. Check ``.available`` and ``.problems``;
        this function does not raise.
    """
    directory = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    results = BenchmarkResults(results_dir=directory)

    if not directory.is_dir():
        results.problems.append(
            f"No results directory at `{directory}`. Generate one with:\n\n"
            "```\npython webapp/export_results.py\n```"
        )
        return results

    results.legacy_files = sorted(
        path.name for path in directory.rglob("*") if path.name in LEGACY_FILENAMES
    )

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        found = sorted(path.name for path in directory.glob("*.csv"))
        detail = f" Found unmanaged CSV files: {', '.join(found)}." if found else ""
        results.problems.append(
            f"`manifest.json` is missing from `{directory}`, so no result file is "
            "trusted as current." + detail + "\n\nRe-export with:\n\n"
            "```\npython webapp/export_results.py\n```"
        )
        return results

    try:
        results.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        results.problems.append(
            f"`manifest.json` could not be parsed: {type(exc).__name__}: {exc}"
        )
        return results

    if not results.manifest.get("experiment_fingerprint"):
        results.problems.append(
            "The manifest records no experiment fingerprint, so these results "
            "cannot be tied to a specific benchmark run."
        )

    declared = results.manifest.get("files") or {}
    for name in REQUIRED_TABLES:
        filename = declared.get(name)
        if not filename:
            results.problems.append(f"The manifest declares no file for the {name} table.")
            continue
        path = directory / filename
        if not path.exists():
            results.problems.append(f"Declared file `{filename}` is missing from `{directory}`.")
            continue
        frame = _read_csv(path, results.problems)
        if frame is not None:
            results.tables[name] = frame

    return results


# --------------------------------------------------------------------------- #
# presentation helpers
# --------------------------------------------------------------------------- #
def maturity_banner(results: BenchmarkResults) -> str:
    """Markdown banner stating how mature the displayed results are."""
    if not results.available:
        return "### No benchmark results loaded"

    manifest = results.manifest
    maturity = manifest.get("maturity", "unknown")
    label = manifest.get("maturity_label") or "Result maturity was not recorded."

    heading = {
        "smoke_test": "Smoke-test results - not intended for model ranking",
        "full_benchmark": "Full benchmark results",
    }.get(maturity, "Benchmark results - maturity not recorded")

    lines = [
        f"### {heading}",
        "",
        label,
        "",
        f"- **Primary metric**: {manifest.get('primary_evaluation_mode', 'unknown')} "
        "scoring (medical knowledge accuracy)",
        "- **Secondary metric**: free generation (answer format and "
        "instruction-following), reported separately in the notebook and not "
        "mixed into the accuracy table above",
        f"- **Run**: `{manifest.get('run_folder', 'unknown')}` "
        f"(fingerprint `{manifest.get('experiment_fingerprint', 'none')}`)",
        f"- **Reference precision for trade-offs**: "
        f"{manifest.get('reference_precision', 'unknown')} of the same family",
    ]
    return "\n".join(lines)


def environment_markdown(results: BenchmarkResults) -> str:
    """Markdown block describing the hardware the benchmark actually ran on."""
    environment = results.environment
    if not environment:
        return (
            "### Benchmark Environment\n\n"
            "No environment metadata was exported with these results."
        )

    def value(key: str, suffix: str = "") -> str:
        raw = environment.get(key)
        if raw in (None, ""):
            return "not recorded"
        return f"{raw}{suffix}"

    rows = [
        ("GPU", value("gpu_name")),
        ("GPU VRAM", value("gpu_total_memory_gib", " GiB")),
        ("CPU RAM", value("cpu_ram_gib", " GiB")),
        ("NVIDIA driver", value("driver_version")),
        ("CUDA", value("cuda_version")),
        ("PyTorch", value("pytorch_version")),
        ("Transformers", value("transformers_version")),
        ("BitsAndBytes", value("bitsandbytes_version")),
        ("Datasets", value("datasets_version")),
        ("Accelerate", value("accelerate_version")),
        ("Python", value("python_version")),
    ]

    lines = [
        "### Benchmark Environment",
        "",
        "Hardware and library versions the benchmark ran on.",
        "",
        "| Item | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {content} |" for name, content in rows]
    lines += [
        "",
        f"*Source: {results.manifest.get('environment_source', 'unknown')}.*",
    ]
    return "\n".join(lines)


def provenance_markdown(results: BenchmarkResults) -> str:
    """Markdown block describing where the displayed numbers came from."""
    manifest = results.manifest
    if not manifest:
        return ""

    commits = manifest.get("resolved_model_commits") or {}
    commit_lines = [f"    - `{family}` @ `{sha[:12]}`" for family, sha in commits.items()]

    lines = [
        "### Result provenance",
        "",
        f"- Export source: `{manifest.get('source', 'unknown')}` "
        f"(`{manifest.get('source_path', 'unknown')}`)",
        f"- Exported at: {manifest.get('exported_at_utc', 'unknown')}",
        f"- Notebook run folder: `{manifest.get('notebook_output_dir', 'unknown')}`",
        f"- Artifact prefix: `{manifest.get('run_prefix', 'unknown')}`",
        f"- Experiment version: `{manifest.get('experiment_version', 'unknown')}`",
        f"- Scoring method: `{manifest.get('scoring_method_version', 'unknown')}`",
        f"- Datasets: {', '.join(manifest.get('benchmarks') or []) or 'unknown'}",
        f"- Random seed: {manifest.get('random_seed', 'unknown')}",
    ]
    if commit_lines:
        lines.append("- Resolved model revisions:")
        lines += commit_lines

    if results.legacy_files:
        lines += [
            "",
            "#### Historical files detected (not displayed as current)",
            "",
            *[f"- `{name}`" for name in results.legacy_files],
            "",
            LEGACY_NOTE,
        ]
    return "\n".join(lines)


def problems_markdown(results: BenchmarkResults) -> str:
    """Markdown block listing loader problems, or an empty string."""
    if not results.problems:
        return ""
    lines = ["### Problems loading benchmark results", ""]
    lines += [f"- {problem}" for problem in results.problems]
    return "\n".join(lines)


def benchmark_row_for(
    results: BenchmarkResults, model_label: str, precision_label: str
) -> dict[str, Any]:
    """Look up the benchmark row for one model configuration.

    Args:
        results: Loaded results.
        model_label: ``Model`` value used in the exported CSVs.
        precision_label: ``Precision`` value used in the exported CSVs.

    Returns:
        A flat dict of benchmark values, empty when the row is not present.
    """
    found: dict[str, Any] = {}
    for name in ("accuracy", "performance", "memory"):
        frame = results.table(name)
        if frame.empty or "Model" not in frame or "Precision" not in frame:
            continue
        match = frame[
            (frame["Model"].astype(str) == model_label)
            & (frame["Precision"].astype(str) == precision_label)
        ]
        if match.empty:
            continue
        record = match.iloc[0].to_dict()
        record.pop("Model", None)
        record.pop("Precision", None)
        found.update(record)
    return found
