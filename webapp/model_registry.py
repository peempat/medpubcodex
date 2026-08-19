"""Model catalogue for the Health Information Assistant demo.

Every repository ID below comes from the benchmark notebook:

* Baseline repositories are ``MODEL_FAMILIES`` in the notebook configuration cell.
* INT8 / INT4-NF4 repositories are the BitsAndBytes checkpoints the notebook
  actually pushed in its "quantized model upload" cell; the recorded upload
  results show all six repositories were created without error.

No repository ID is invented here. The quantized repositories were pushed with
``HF_PRIVATE = True``, so a Hugging Face token with access to the
``pupupapapa`` namespace is required to load them.
"""

from __future__ import annotations

from dataclasses import dataclass

# Namespace used by the notebook uploader (``HF_NAMESPACE or whoami["name"]``).
QUANTIZED_NAMESPACE = "pupupapapa"


@dataclass(frozen=True)
class ModelEntry:
    """One selectable model configuration."""

    key: str
    label: str
    family: str
    """Family name as used by the benchmark notebook (e.g. ``MedGemma``)."""
    precision: str
    """Precision name as used by the benchmark notebook (e.g. ``INT4-NF4``)."""
    repo_id: str
    base_model_id: str
    revision: str | None = None
    is_private: bool = False
    requires_cuda: bool = False
    loads_prequantized: bool = False
    """True when the checkpoint already carries a BitsAndBytes quantization config."""
    benchmark_row: str = ""
    """``Model`` value used in the exported benchmark CSVs, for cross-referencing."""
    benchmark_precision: str = ""
    """``Precision`` value used in the exported benchmark CSVs."""
    notes: str = ""


# Benchmark-notebook family name -> label used in the exported report tables.
FAMILY_REPORT_LABELS = {
    "Gemma4-E4B": "Gemma 4 E4B",
    "Gemma4-12B": "Gemma 4 12B",
    "MedGemma": "MedGemma 1.5 4B",
}

# Benchmark-notebook precision -> label used in the exported report tables.
PRECISION_REPORT_LABELS = {
    "Baseline": "BF16/FP16",
    "INT8": "INT8",
    "INT4-NF4": "INT4",
}

_BASE_MODELS = {
    "Gemma4-E4B": "google/gemma-4-E4B-it",
    "Gemma4-12B": "google/gemma-4-12B-it",
    "MedGemma": "google/medgemma-1.5-4b-it",
}

# Repository names produced by the notebook uploader:
#   f"{model_id.split('/')[-1].lower()}-{precision.lower().replace('-nf4','')}-bnb"
_QUANTIZED_REPOS = {
    ("Gemma4-E4B", "INT8"): "gemma-4-e4b-it-int8-bnb",
    ("Gemma4-E4B", "INT4-NF4"): "gemma-4-e4b-it-int4-bnb",
    ("Gemma4-12B", "INT8"): "gemma-4-12b-it-int8-bnb",
    ("Gemma4-12B", "INT4-NF4"): "gemma-4-12b-it-int4-bnb",
    ("MedGemma", "INT8"): "medgemma-1.5-4b-it-int8-bnb",
    ("MedGemma", "INT4-NF4"): "medgemma-1.5-4b-it-int4-bnb",
}


def _build_registry() -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for family, base_id in _BASE_MODELS.items():
        family_label = FAMILY_REPORT_LABELS[family]
        for precision in ("Baseline", "INT8", "INT4-NF4"):
            report_precision = PRECISION_REPORT_LABELS[precision]
            if precision == "Baseline":
                entries.append(
                    ModelEntry(
                        key=f"{family}:{precision}",
                        label=f"{family_label} - Baseline BF16",
                        family=family,
                        precision=precision,
                        repo_id=base_id,
                        base_model_id=base_id,
                        is_private=False,
                        # Gated upstream repository; 4B-12B multimodal weights are
                        # not practical to serve on CPU, so a GPU is required here too.
                        requires_cuda=True,
                        loads_prequantized=False,
                        benchmark_row=family_label,
                        benchmark_precision=report_precision,
                        notes=(
                            "Unquantized upstream model. Highest VRAM cost; served here "
                            "only as the reference point for the quantized entries."
                        ),
                    )
                )
                continue

            repo_name = _QUANTIZED_REPOS[(family, precision)]
            entries.append(
                ModelEntry(
                    key=f"{family}:{precision}",
                    label=f"{family_label} - {report_precision}",
                    family=family,
                    precision=precision,
                    repo_id=f"{QUANTIZED_NAMESPACE}/{repo_name}",
                    base_model_id=base_id,
                    is_private=True,
                    requires_cuda=True,
                    loads_prequantized=True,
                    benchmark_row=family_label,
                    benchmark_precision=report_precision,
                    notes=(
                        f"BitsAndBytes {report_precision} checkpoint pushed by the benchmark "
                        "notebook. Private repository; requires a Hugging Face token with "
                        "read access. BitsAndBytes requires a CUDA GPU."
                    ),
                )
            )
    return entries


MODELS: list[ModelEntry] = _build_registry()
MODELS_BY_KEY: dict[str, ModelEntry] = {entry.key: entry for entry in MODELS}
MODELS_BY_LABEL: dict[str, ModelEntry] = {entry.label: entry for entry in MODELS}

# MedGemma INT4-NF4 is the default: the exported benchmark records it as the
# lowest-VRAM configuration measured (4.30 GiB peak allocated), and it is the
# medically tuned family. Note that the benchmark's highest *constrained*
# accuracy was Gemma 4 E4B INT4 - see the Benchmark Explorer tab.
DEFAULT_MODEL_KEY = "MedGemma:INT4-NF4"


def default_model() -> ModelEntry:
    return MODELS_BY_KEY[DEFAULT_MODEL_KEY]


def get_model(key_or_label: str) -> ModelEntry:
    """Resolve a dropdown label or registry key to a :class:`ModelEntry`.

    Raises:
        KeyError: if the selection is not a known model.
    """
    if key_or_label in MODELS_BY_KEY:
        return MODELS_BY_KEY[key_or_label]
    if key_or_label in MODELS_BY_LABEL:
        return MODELS_BY_LABEL[key_or_label]
    raise KeyError(f"Unknown model selection: {key_or_label!r}")


def choice_labels() -> list[str]:
    """Dropdown choices, quantized entries first (cheapest to run)."""
    quantized = [entry.label for entry in MODELS if entry.precision != "Baseline"]
    baseline = [entry.label for entry in MODELS if entry.precision == "Baseline"]
    return quantized + baseline
