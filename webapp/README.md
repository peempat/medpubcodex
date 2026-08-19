# Quantized Medical LLM — Health Information Assistant

A small Gradio research demo built on top of the MedPubCodex quantization
benchmark. It takes the quantized checkpoints the benchmark produced and puts
them behind a health-information interface, so the chain

```
Benchmark → Quantization → Model comparison → Model selection → Practical use case
```

is visible end to end in one application.

**This is a research demo, not a medical product.**

---

## 1. What this Web App demonstrates

The benchmark answers a research question: *what does INT8 / INT4 quantization
cost you on medical QA, and what does it buy you in memory and speed?* This app
answers the follow-up question: *what does the winning configuration actually
feel like to serve?*

Concretely it shows:

- A quantized checkpoint being loaded and generating answers, with **live**
  latency, tokens/sec and peak-VRAM numbers measured on the machine you run it on.
- The **benchmark** accuracy / performance / memory / trade-off tables that
  justified the model choice, loaded from exported result files rather than
  retyped into the code.
- Both sets of numbers side by side, clearly separated, because they were
  measured on different hardware.

## 2. Relationship to the benchmark project

| | Benchmark notebook | This Web App |
| --- | --- | --- |
| File | `gemma_medical_quantization_benchmark.ipynb` | `webapp/` |
| Purpose | Controlled evaluation | Demonstration of the outcome |
| Metrics | Constrained MCQ accuracy, latency, VRAM over 100 questions/dataset | One live generation per request |
| Hardware | NVIDIA A100-SXM4-40GB (Colab) | Whatever you run it on |
| Models | 3 families × 3 precisions | The same 9 configurations, selectable |

The notebook is the source of truth. This app **never** recomputes a benchmark
number and never modifies the notebook. `export_results.py` copies the
notebook's own report tables into `webapp/results/current/`.

The current results are a **smoke test**: the notebook ran with
`SMOKE_TEST_ONLY = True` and `SMOKE_SAMPLES_PER_DATASET = 100`, so the tables
cover 100 questions per dataset. The app labels them that way and states they
are not intended for final model ranking.

## 3. Supported use cases

| Use case | Example | What it does |
| --- | --- | --- |
| General Health Q&A | *What is high blood pressure?* | Broad educational answers |
| Explain Medical Term | *What does hypertension mean?* | Plain-language definitions |
| Disease Information | *What is diabetes mellitus?* | What the condition is, common symptoms, common risk factors, general health information |
| Nutrition Information | *What foods are high in fiber?* | General educational nutrition information |

Each answer is generated from the **current question alone**. The conversation
stays visible in the UI, but earlier turns are not fed back into the model:
multi-turn behaviour was not part of the benchmark and has not been evaluated.

## 4. Medical safety scope

> **For research and educational purposes only.**
> This application does not provide medical diagnosis and does not replace a
> qualified healthcare professional.

The app will not:

- tell you what condition you have, or rank likely diagnoses for your case
- give medication doses, quantities, schedules, or drug choices
- advise starting, stopping, or changing a medicine
- write a treatment plan or a personalised therapeutic diet

Two layers enforce this:

1. **Prompting** (`prompts.py`) — the system prompt states the scope for every
   turn and asks for the same language the user wrote in.
2. **A deterministic rule layer** (`safety.py`) — runs outside the model:
   - Before generation, `check_request()` classifies the question. Diagnosis,
     dosing, and treatment-plan requests are **redirected**: the user still gets
     general information, with a scope note and an extra prompt constraint.
     Questions describing potentially urgent symptoms, and messages suggesting
     self-harm, **bypass the model entirely** and return care-seeking guidance.
   - After generation, `review_answer()` looks for direct diagnostic assertions
     ("you have X") and attaches a correction notice.

**This guardrail is a simple keyword and pattern filter written for a demo. It
is not a clinically validated safety system.** It will miss phrasings it was not
written for and will sometimes redirect a harmless question. Do not present it
as a medical safeguard.

Example of intended behaviour:

```
User: I have headache and dizziness. What disease do I have?

App:  > Scope note. This demo does not identify what condition you have...
      [general information about non-specific symptoms and their many possible
       causes, plus a recommendation to see a healthcare professional]
```

## 5. Required hardware

**Benchmark Explorer tab**: no GPU needed. Runs anywhere Python and Gradio run.

**Health Information Assistant tab**: needs a **CUDA GPU**. The INT8/INT4
entries are BitsAndBytes checkpoints, and BitsAndBytes requires CUDA. The
unquantized baselines are 4B–12B multimodal models and are not practical on CPU,
so the app requires a GPU for every model entry and says so clearly when there
is none.

Peak VRAM measured by the benchmark, as a sizing guide:

| Configuration | Peak VRAM |
| --- | --- |
| MedGemma 1.5 4B INT4 | 4.30 GiB |
| MedGemma 1.5 4B INT8 | 5.95 GiB |
| Gemma 4 12B INT4 | 9.85 GiB |
| Gemma 4 E4B INT4 | 10.80 GiB |
| Gemma 4 E4B INT8 | 12.80 GiB |
| Gemma 4 12B INT8 | 14.79 GiB |
| MedGemma 1.5 4B BF16 | 9.30 GiB |
| Gemma 4 E4B BF16 | 16.92 GiB |
| Gemma 4 12B BF16 | 24.98 GiB |

Add headroom for the KV cache and activations; these are peak allocated figures
from the benchmark run, not a guaranteed footprint on your machine.

## 6. Installation

**Windows PowerShell**

```powershell
cd C:\path\to\medpubcodex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r webapp\requirements.txt
```

**Linux / macOS**

```bash
cd /path/to/medpubcodex
python -m venv .venv
source .venv/bin/activate
pip install -r webapp/requirements.txt
```

`requirements.txt` lists a generic `torch`. To run the model tab you need a
**CUDA build**:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 7. Hugging Face token setup

A token is required because:

- `google/gemma-4-*` and `google/medgemma-*` are **gated** — accept the licence
  on the model page with the same account first;
- the quantized checkpoints were pushed as **private** repositories.

Create a read token at <https://huggingface.co/settings/tokens>, then:

**Windows PowerShell**

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

**Linux / macOS**

```bash
export HF_TOKEN=hf_your_token_here
```

**Or via a file** (preferred for repeated runs):

```bash
cp webapp/.env.example webapp/.env   # PowerShell: Copy-Item webapp\.env.example webapp\.env
# then edit webapp/.env
```

`webapp/.env` is git-ignored. **Never commit a token.** `HF_TOKEN`,
`HUGGINGFACE_HUB_TOKEN` and `HUGGINGFACEHUB_API_TOKEN` are all read; the app
reports only whether a token is *present*, never its value.

## 8. How to run locally

```bash
python webapp/app.py
```

Then open <http://127.0.0.1:7860>.

Override the binding with `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT`, or
`GRADIO_SHARE=true`.

Run the tests:

```bash
python -m pytest webapp/tests -q
```

## 9. How benchmark results are loaded

```
notebook run  ──►  export_results.py  ──►  webapp/results/current/  ──►  benchmark_results.py  ──►  Benchmark Explorer
```

`webapp/results/current/` holds four CSVs plus a `manifest.json`:

| File | Contents |
| --- | --- |
| `accuracy_comparison.csv` | Table 1 — constrained accuracy per dataset |
| `performance_comparison.csv` | Table 2 — tokens/sec, latency, P95, load time |
| `memory_comparison.csv` | Table 3 — peak VRAM, reserved VRAM, peak RAM |
| `quantization_tradeoff.csv` | Table 4 — accuracy drop, speedup, VRAM reduction |
| `manifest.json` | Run identity, fingerprint, maturity, benchmark environment |

The exporter has two modes:

```bash
# Default: read the tables out of the executed notebook's stored cell outputs.
python webapp/export_results.py

# Preferred when the run folder is reachable: copy the notebook's own CSVs.
python webapp/export_results.py --from-run-dir "/content/drive/MyDrive/gemma4_medgemma_benchmark_results/runs/corrected_pipeline_v6_20260819"
```

Both modes produce the same filenames. Neither recomputes any metric.

**Guards against stale results**

- `manifest.json` is **required**. Loose CSVs with no manifest are refused,
  because there would be no record of which run they came from.
- Files whose names carry no experiment fingerprint (the earlier
  `clean_three_model_*` outputs) are reported as **historical** under *Result
  provenance* and are never merged into the current tables. That earlier run
  contained an anomalous MedMCQA result which the corrected pipeline replaced;
  the current export shows no zero-valued accuracy, and a test asserts it.
- Missing or unreadable files are listed in the UI rather than silently skipped.

**Primary vs secondary metric.** The Accuracy table is *constrained scoring*
only — the notebook's primary medical-knowledge metric. Free-generation results
measure answer format and instruction-following and are a separate secondary
metric; they are not shown as accuracy here.

## 10. Benchmark metrics vs live demo metrics

These are never mixed.

| | Benchmark metrics | Live demo metrics |
| --- | --- | --- |
| Where | Benchmark Explorer tab, and the model card | "Live demo metrics" panel |
| Source | Controlled notebook run | This machine, this request |
| Hardware | NVIDIA A100-SXM4-40GB, CUDA 12.8, PyTorch 2.11.0+cu128, transformers 5.14.1, bitsandbytes 0.49.2 | Reported under *Current Demo Runtime* |
| Sample size | 100 questions per dataset, repeated timing runs | One generation |
| Comparable? | With each other | With each other |

A single generation on different hardware says nothing about the benchmark
ranking, and the benchmark says nothing about what your GPU will do. The app
shows both environments explicitly so the difference stays visible.

---

## Files

| File | Role |
| --- | --- |
| `app.py` | Gradio UI, both tabs, chat callback |
| `inference.py` | Lazy model load/unload, generation, live metrics, error translation |
| `model_registry.py` | The 9 model configurations and their real repository IDs |
| `prompts.py` | System prompt and the four use cases |
| `safety.py` | Deterministic scope layer (pre- and post-generation) |
| `benchmark_results.py` | Result loading, validation, provenance, presentation |
| `export_results.py` | Adapter from notebook / run folder to `results/current/` |
| `results/current/` | Exported benchmark tables and manifest |
| `tests/` | Safety, benchmark loader, model registry, error handling |

## Model selection

The default is **MedGemma 1.5 4B INT4** (`pupupapapa/medgemma-1.5-4b-it-int4-bnb`)
— the lowest-VRAM configuration in the benchmark (4.30 GiB) and the medically
tuned family, so it is the one most likely to load on demo hardware.

Note that on the benchmark's *constrained* accuracy metric the highest score was
**Gemma 4 E4B INT4** (0.4233 average, 10.80 GiB), which is also selectable. That
metric is multiple-choice medical QA, not open health-information quality, so it
does not by itself establish which model answers this app's questions better.

## Known limitations

- **Research demo only.** Not a medical device, not a clinical tool.
- **Not a diagnostic system.** It will not tell anyone what condition they have.
- **Model answers may be incorrect.** These are quantized general-purpose and
  medically tuned LLMs; they produce fluent text that can be wrong.
- **Benchmark accuracy does not establish clinical safety.** Scoring 42% on
  constrained multiple-choice medical QA says nothing about whether an answer is
  safe to act on.
- **The safety layer is a simple pattern filter**, not a validated safeguard.
- **Demo hardware performance will differ from benchmark hardware.** Live
  metrics and benchmark metrics are not comparable.
- **Current results are a smoke test** (n=100 per dataset), explicitly not a
  final ranking.
- **The quantized repositories are private and the base models are gated**, so
  the model tab does not work without an authorised token.
