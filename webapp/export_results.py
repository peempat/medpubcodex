"""Export benchmark artifacts from the benchmark notebook into ``webapp/results/current``.

The benchmark notebook writes every artifact into a Google Drive run folder that is
not part of this repository. The Web App must not hardcode benchmark numbers, so
this adapter produces a small, self-contained result set that ``benchmark_results``
can load.

Two sources are supported:

``--from-run-dir <path>``
    Preferred. Copies the report CSVs that the notebook itself wrote
    (``accuracy_comparison.csv`` and friends) out of a completed run folder.

``--from-notebook`` (default)
    Fallback used when the Drive run folder is not reachable. Reads the report
    tables straight out of the executed notebook stored cell outputs. No
    scientific logic is re-implemented here: the tables are taken verbatim from
    the outputs the notebook already produced.

Nothing in this module recomputes accuracy, latency or memory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = REPO_ROOT / "gemma_medical_quantization_benchmark.ipynb"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "current"

# Heading emitted by the notebook -> logical table name used by the Web App.
TABLE_HEADINGS = {
    "Table 1 - Primary Constrained Medical Accuracy": "accuracy",
    "Table 2 - Inference Performance": "performance",
    "Table 3 - Memory Efficiency": "memory",
    "Table 4 - Quantization Trade-off": "tradeoff",
}

# Logical table name -> filename. These match the filenames the notebook writes
# in section 16, so ``--from-run-dir`` and ``--from-notebook`` agree.
TABLE_FILENAMES = {
    "accuracy": "accuracy_comparison.csv",
    "performance": "performance_comparison.csv",
    "memory": "memory_comparison.csv",
    "tradeoff": "quantization_tradeoff.csv",
}

# Files whose names carry no experiment fingerprint belong to superseded runs.
LEGACY_FILENAME_PATTERNS = (
    "clean_three_model_raw_predictions.csv",
    "clean_three_model_benchmark_summary.csv",
    "clean_three_model_performance_summary.csv",
    "clean_three_model_memory_summary.csv",
)


class ExportError(RuntimeError):
    """Raised when the notebook or run folder does not contain usable results."""


# --------------------------------------------------------------------------- #
# notebook helpers
# --------------------------------------------------------------------------- #
def load_notebook(path: Path) -> dict:
    if not path.exists():
        raise ExportError(f"Notebook not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def stream_text(cell: dict) -> str:
    parts = []
    for output in cell.get("outputs", []) or []:
        if output.get("output_type") == "stream":
            parts.append("".join(output.get("text", [])))
        elif output.get("output_type") in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/plain" in data:
                parts.append("".join(data["text/plain"]))
    return "\n".join(parts)


def literal_assignments(source: str, names: set[str]) -> dict[str, Any]:
    """Read simple literal assignments out of a cell without executing it."""
    found: dict[str, Any] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    return found


def markdown_table_outputs(cell: dict) -> list[tuple[str, dict]]:
    """Pair each ``display(Markdown(...))`` heading with the DataFrame that follows it."""
    pairs: list[tuple[str, dict]] = []
    heading = ""
    for output in cell.get("outputs", []) or []:
        data = output.get("data", {})
        if "text/markdown" in data:
            heading = "".join(data["text/markdown"]).strip().lstrip("#").strip()
            continue
        payload = data.get("application/vnd.microsoft.datawrangler.viewer.v0+json")
        if payload is not None:
            pairs.append((heading, payload))
    return pairs


def datawrangler_to_rows(payload: dict) -> tuple[list[str], list[list[str]]]:
    """Convert a Data Wrangler payload into ``(header, rows)``, dropping the index column."""
    columns = [column["name"] for column in payload.get("columns", [])]
    rows = [list(row) for row in payload.get("rows", [])]
    if columns and columns[0] == "index":
        columns = columns[1:]
        rows = [row[1:] for row in rows]
    return columns, rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# metadata extraction
# --------------------------------------------------------------------------- #
def extract_environment(notebook: dict) -> dict[str, Any]:
    """Recover the benchmark hardware/software environment from executed outputs.

    The notebook stores the same values in ``<run_prefix>_config.json`` under
    ``environment``. That file lives on Drive, so the printed runtime check and
    version banner are used instead. Values are never invented: anything that
    was not printed stays ``None``.
    """
    environment: dict[str, Any] = {
        "gpu_name": None,
        "gpu_total_memory_gib": None,
        "driver_version": None,
        "cuda_version": None,
        "pytorch_version": None,
        "python_version": None,
        "transformers_version": None,
        "datasets_version": None,
        "accelerate_version": None,
        "bitsandbytes_version": None,
        "cpu_ram_gib": None,
    }

    text = "\n".join(stream_text(cell) for cell in notebook["cells"])

    patterns = {
        "gpu_name": r"^GPU:\s*(.+)$",
        "gpu_total_memory_gib": r"^Total GPU memory:\s*([\d.]+)\s*GiB",
        "cuda_version": r"^CUDA reported by PyTorch:\s*(.+)$",
        "pytorch_version": r"^PyTorch:\s*(.+)$",
        "python_version": r"^Python:\s*(.+)$",
        "transformers_version": r"^Transformers:\s*(.+)$",
        "datasets_version": r"^Datasets:\s*(.+)$",
        "accelerate_version": r"^Accelerate:\s*(.+)$",
        "bitsandbytes_version": r"^BitsAndBytes:\s*(.+)$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            environment[key] = match.group(1).strip()

    if environment["gpu_total_memory_gib"]:
        environment["gpu_total_memory_gib"] = float(environment["gpu_total_memory_gib"])

    # nvidia-smi CSV line: "NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07"
    smi = re.search(
        r"^(NVIDIA [^,\n]+),\s*(\d+)\s*MiB,\s*([\d.]+)\s*$", text, flags=re.MULTILINE
    )
    if smi:
        environment["driver_version"] = smi.group(3)

    return environment


def extract_run_identity(notebook: dict) -> dict[str, Any]:
    """Recover run prefix, experiment fingerprint, resolved model commits and scope."""
    identity: dict[str, Any] = {
        "run_prefix": None,
        "experiment_fingerprint": None,
        "run_folder": None,
        "notebook_output_dir": None,
        "resolved_model_commits": {},
    }

    text = "\n".join(stream_text(cell) for cell in notebook["cells"])

    prefix_match = re.search(r"Artifacts written with prefix '([A-Za-z0-9_]+?)_'", text)
    if prefix_match:
        run_prefix = prefix_match.group(1)
        identity["run_prefix"] = run_prefix
        fingerprint = run_prefix.rsplit("_", 1)[-1]
        if re.fullmatch(r"[0-9a-f]{8,}", fingerprint):
            identity["experiment_fingerprint"] = fingerprint

    folder_match = re.search(r"Live Drive folder:\s*(.+)$", text, flags=re.MULTILINE)
    if folder_match:
        output_dir = folder_match.group(1).strip()
        identity["notebook_output_dir"] = output_dir
        identity["run_folder"] = output_dir.rstrip("/").rsplit("/", 1)[-1]

    for family, commit in re.findall(
        r"^\s{2}(\S+)\s+([0-9a-f]{40})\s*$", text, flags=re.MULTILINE
    ):
        identity["resolved_model_commits"][family] = commit

    return identity


def extract_scope(notebook: dict) -> dict[str, Any]:
    """Read the benchmark scope constants so the UI can label result maturity."""
    wanted = {
        "MODEL_FAMILIES",
        "RUN_PRECISIONS",
        "BENCHMARKS",
        "EVALUATION_MODES",
        "PRIMARY_EVALUATION_MODE",
        "MAX_SAMPLES_PER_DATASET",
        "SMOKE_TEST_ONLY",
        "RUN_SMOKE_TEST",
        "SMOKE_SAMPLES_PER_DATASET",
        "FREE_GENERATION_MAX_SAMPLES_PER_DATASET",
        "REFERENCE_PRECISION",
        "EXPERIMENT_VERSION",
        "SCORING_METHOD_VERSION",
        "RANDOM_SEED",
    }
    values: dict[str, Any] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        values.update(literal_assignments(cell_source(cell), wanted))

    smoke_only = bool(values.get("SMOKE_TEST_ONLY"))
    if smoke_only:
        samples = values.get("SMOKE_SAMPLES_PER_DATASET")
        maturity = "smoke_test"
        label = (
            f"Smoke-test / preliminary results (n={samples} questions per dataset) "
            "- exploratory only, not intended for final model ranking"
        )
    else:
        samples = values.get("MAX_SAMPLES_PER_DATASET")
        maturity = "full_benchmark"
        label = f"Full benchmark run (n={samples} questions per dataset)"

    return {
        "maturity": maturity,
        "maturity_label": label,
        "constrained_samples_per_dataset": samples,
        "free_generation_samples_per_dataset": values.get(
            "FREE_GENERATION_MAX_SAMPLES_PER_DATASET"
        ),
        "primary_evaluation_mode": values.get("PRIMARY_EVALUATION_MODE"),
        "evaluation_modes": values.get("EVALUATION_MODES"),
        "benchmarks": values.get("BENCHMARKS"),
        "reference_precision": values.get("REFERENCE_PRECISION"),
        "experiment_version": values.get("EXPERIMENT_VERSION"),
        "scoring_method_version": values.get("SCORING_METHOD_VERSION"),
        "random_seed": values.get("RANDOM_SEED"),
        "model_families": values.get("MODEL_FAMILIES"),
        "precisions": [
            name for name, enabled in (values.get("RUN_PRECISIONS") or {}).items() if enabled
        ],
    }


# --------------------------------------------------------------------------- #
# export modes
# --------------------------------------------------------------------------- #
def export_from_notebook(notebook_path: Path, out_dir: Path) -> dict[str, Any]:
    notebook = load_notebook(notebook_path)

    tables: dict[str, tuple[list[str], list[list[str]]]] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for heading, payload in markdown_table_outputs(cell):
            for known_heading, logical in TABLE_HEADINGS.items():
                if heading.startswith(known_heading) and logical not in tables:
                    tables[logical] = datawrangler_to_rows(payload)

    missing = sorted(set(TABLE_FILENAMES) - set(tables))
    if missing:
        raise ExportError(
            "The notebook has no stored output for: "
            + ", ".join(missing)
            + ". Run the notebook section 16 report cells, or use --from-run-dir."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for logical, (header, rows) in tables.items():
        filename = TABLE_FILENAMES[logical]
        write_csv(out_dir / filename, header, rows)
        written[logical] = filename

    manifest = build_manifest(
        source="notebook_outputs",
        source_path=notebook_path,
        notebook=notebook,
        written=written,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def export_from_run_dir(run_dir: Path, notebook_path: Path, out_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise ExportError(f"Run folder not found: {run_dir}")

    notebook = load_notebook(notebook_path)
    identity = extract_run_identity(notebook)
    fingerprint = identity.get("experiment_fingerprint")

    stale = [name for name in LEGACY_FILENAME_PATTERNS if (run_dir / name).exists()]

    missing = [
        filename for filename in TABLE_FILENAMES.values() if not (run_dir / filename).exists()
    ]
    if missing:
        raise ExportError(
            f"Run folder {run_dir} is missing report CSVs: {', '.join(missing)}. "
            "Run the notebook section 16 report cells first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for logical, filename in TABLE_FILENAMES.items():
        shutil.copyfile(run_dir / filename, out_dir / filename)
        written[logical] = filename

    # The fingerprinted config carries the authoritative environment block.
    config_name = f"{identity['run_prefix']}_config.json" if identity["run_prefix"] else None
    config_copied = None
    if config_name and (run_dir / config_name).exists():
        shutil.copyfile(run_dir / config_name, out_dir / "run_config.json")
        config_copied = config_name

    manifest = build_manifest(
        source="run_directory",
        source_path=run_dir,
        notebook=notebook,
        written=written,
    )
    manifest["run_config_file"] = config_copied
    manifest["ignored_legacy_files"] = stale
    if config_copied:
        config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
        if config.get("environment"):
            manifest["environment"] = config["environment"]
            manifest["environment_source"] = f"{config_copied} (environment block)"
        if config.get("experiment_fingerprint"):
            manifest["experiment_fingerprint"] = config["experiment_fingerprint"]
    if fingerprint and manifest.get("experiment_fingerprint") != fingerprint:
        manifest["fingerprint_warning"] = (
            "Run folder fingerprint does not match the notebook last executed run."
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_manifest(
    source: str, source_path: Path, notebook: dict, written: dict[str, str]
) -> dict[str, Any]:
    identity = extract_run_identity(notebook)
    scope = extract_scope(notebook)
    try:
        # Keep the manifest free of absolute local paths when the source is in-repo.
        recorded_path = source_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        recorded_path = str(source_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_path": recorded_path,
        "environment_source": "notebook runtime-check and version-banner outputs",
        "environment": extract_environment(notebook),
        "files": written,
    }
    manifest.update(identity)
    manifest.update(scope)
    return manifest


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export benchmark results for the Web App.")
    parser.add_argument(
        "--from-run-dir",
        type=Path,
        default=None,
        help="Completed notebook run folder (for example a Google Drive run directory).",
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    try:
        if args.from_run_dir is not None:
            manifest = export_from_run_dir(args.from_run_dir, args.notebook, args.out_dir)
        else:
            manifest = export_from_notebook(args.notebook, args.out_dir)
    except ExportError as exc:
        print(f"Export failed: {exc}")
        return 1

    print(f"Exported {len(manifest['files'])} tables to {args.out_dir}")
    print(f"  source       : {manifest['source']}")
    print(f"  run prefix   : {manifest.get('run_prefix')}")
    print(f"  fingerprint  : {manifest.get('experiment_fingerprint')}")
    print(f"  maturity     : {manifest.get('maturity')}")
    print(f"  benchmark GPU: {manifest['environment'].get('gpu_name')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
