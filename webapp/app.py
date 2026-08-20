"""Quantized Medical LLM - Health Information Assistant (research demo).

Gradio front end for the MedPubCodex quantization benchmark. Two tabs:

1. Health Information Assistant - single-turn health-information answers from a
   quantized checkpoint, with live runtime metrics.
2. Benchmark Explorer - the benchmark results that motivated the model choice.

Run with::

    python webapp/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import gradio as gr

import benchmark_results as br
import model_registry
import prompts
import safety
from inference import RUNTIME, InferenceError, describe_runtime

APP_TITLE = "Quantized Medical LLM - Health Information Assistant"

HEADER_MARKDOWN = f"""
# {APP_TITLE}

**Research Demo** | Quantized Gemma 4 / MedGemma checkpoints from the MedPubCodex
quantization benchmark.

{safety.DISCLAIMER}
"""

SCOPE_MARKDOWN = """
### What this demo does and does not do

**Does**: general health information, plain-language explanations of medical
terms, general overviews of named conditions, general nutrition information.

**Does not**: tell you what condition you have, give medication doses or drug
choices, write treatment plans, or replace a clinical assessment.

A small rule-based filter runs outside the model to keep questions inside that
scope. It is a simple pattern filter written for this demo, not a clinically
validated safety system.
"""


# --------------------------------------------------------------------------- #
# shared state loaded once at start-up
# --------------------------------------------------------------------------- #
RESULTS = br.load_results()


def _load_dotenv_if_present() -> None:
    """Read ``webapp/.env`` into the environment when python-dotenv is installed."""
    env_path = WEBAPP_DIR / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # Minimal fallback so the demo works without the optional dependency.
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv_if_present()


# --------------------------------------------------------------------------- #
# metric rendering
# --------------------------------------------------------------------------- #
IDLE_METRICS = (
    "### Live demo metrics\n\n"
    "_Measured on this machine, for this request._\n\n"
    "Ask a question to populate this panel."
)


def render_live_metrics(metrics) -> str:
    """Markdown for the live metrics panel."""
    vram = (
        f"{metrics.peak_vram_gib:.2f} GiB"
        if metrics.peak_vram_gib is not None
        else "not available (no CUDA device)"
    )
    lines = [
        "### Live demo metrics",
        "",
        "_Measured on this machine, for this request. Not comparable with the "
        "benchmark numbers in the Benchmark Explorer tab._",
        "",
        f"- **Model**: {metrics.model_label}",
        f"- **Precision**: {metrics.precision}",
        f"- **Checkpoint**: `{metrics.repo_id}`",
        f"- **Device**: {metrics.device}",
        "",
        f"- **Generation latency**: {metrics.latency_seconds:.2f} s",
        f"- **Generated tokens**: {metrics.generated_tokens}",
        f"- **Tokens/sec**: {metrics.tokens_per_second:.1f}",
        f"- **Prompt tokens**: {metrics.prompt_tokens}",
        f"- **Peak VRAM**: {vram}",
    ]
    if metrics.was_freshly_loaded and metrics.load_time_seconds is not None:
        lines.append(f"- **Model load time**: {metrics.load_time_seconds:.1f} s (loaded this turn)")
    return "\n".join(lines)


# Benchmark columns worth surfacing on the model card, with their units.
BENCHMARK_CARD_FIELDS = (
    ("Average Constrained Accuracy", ""),
    ("Tokens/sec", " tok/s"),
    ("Mean Latency", " s"),
    ("Peak VRAM", " GiB"),
    ("Load Time", " s"),
)


def render_model_card(model_label: str) -> str:
    """Markdown describing the selected model, including its benchmark row."""
    try:
        entry = model_registry.get_model(model_label)
    except KeyError:
        return f"Unknown model selection: `{model_label}`."

    lines = [
        f"**Checkpoint**: `{entry.repo_id}`",
        f"**Base model**: `{entry.base_model_id}`",
        f"**Precision**: {entry.precision}",
        "",
        entry.notes,
    ]

    row = br.benchmark_row_for(RESULTS, entry.benchmark_row, entry.benchmark_precision)
    if row:
        lines += [
            "",
            "**Benchmark metrics for this configuration** "
            "_(controlled notebook run, different hardware)_",
            "",
        ]
        for key, unit in BENCHMARK_CARD_FIELDS:
            if key in row and row[key] != "":
                lines.append(f"- {key}: {row[key]}{unit}")
    else:
        lines += ["", "_No benchmark row was found for this configuration._"]
    return "\n".join(lines)


def render_demo_runtime() -> str:
    """Markdown describing the machine the demo is running on right now."""
    info = describe_runtime()
    rows = [
        ("Platform", info["platform"]),
        ("Python", info["python_version"]),
        ("PyTorch", info["torch_version"] or "not installed"),
        ("CUDA available", "yes" if info["cuda_available"] else "no"),
        ("CUDA (PyTorch build)", info["cuda_version"] or "none"),
        ("GPU", info["gpu_name"] or "none detected"),
        (
            "GPU VRAM",
            f"{info['gpu_total_memory_gib']} GiB" if info["gpu_total_memory_gib"] else "n/a",
        ),
        ("CPU RAM", f"{info['cpu_ram_gib']} GiB" if info["cpu_ram_gib"] else "not detected"),
        ("Transformers", info["transformers_version"] or "not installed"),
        ("BitsAndBytes", info["bitsandbytes_version"] or "not installed"),
        ("HF token detected", "yes" if info["hf_token_present"] else "no"),
    ]
    lines = [
        "### Current Demo Runtime",
        "",
        "The machine serving this page. It may differ from the benchmark "
        "hardware above, so live metrics and benchmark metrics are not "
        "comparable.",
        "",
        "| Item | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value} |" for name, value in rows]

    if not info["cuda_available"]:
        lines += [
            "",
            "> **No CUDA device detected.** The BitsAndBytes INT8/INT4 checkpoints "
            "cannot be loaded here. The Benchmark Explorer works without a GPU.",
        ]
    if not info["hf_token_present"]:
        lines += [
            "",
            "> **No Hugging Face token detected.** The quantized checkpoints are "
            "private and the Gemma repositories are gated, so loading will fail "
            "until a token is set. See `webapp/.env.example`.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# chat callback
# --------------------------------------------------------------------------- #
def respond(question: str, history: list, model_label: str, use_case_label: str):
    """Handle one turn: scope check, generation, response review.

    Generation is single-turn; ``history`` is displayed but not fed to the model.
    """
    history = list(history or [])
    text = (question or "").strip()

    scope = safety.check_request(text)

    if scope.is_blocked:
        shown = text if text else "(empty message)"
        history.append({"role": "user", "content": shown})
        history.append({"role": "assistant", "content": scope.blocked_response})
        return history, "", IDLE_METRICS

    try:
        entry = model_registry.get_model(model_label)
    except KeyError:
        history.append({"role": "user", "content": text})
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"`{model_label}` is not a known model configuration. "
                    "Pick one from the Model dropdown."
                ),
            }
        )
        return history, "", IDLE_METRICS

    try:
        use_case = prompts.get_use_case(use_case_label)
    except KeyError:
        use_case = prompts.USE_CASES_BY_KEY[prompts.DEFAULT_USE_CASE_KEY]

    history.append({"role": "user", "content": text})

    def with_notice(body: str) -> str:
        """Keep the scope note visible even when generation could not run."""
        return f"{scope.notice}\n\n{body}" if scope.notice else body

    messages = prompts.build_messages(text, use_case, scope.guidance)
    try:
        answer, metrics = RUNTIME.generate(messages, entry)
    except InferenceError as exc:
        history.append(
            {"role": "assistant", "content": with_notice(f"**Could not answer.**\n\n{exc}")}
        )
        return history, "", IDLE_METRICS
    except Exception as exc:  # last-resort guard so the UI never dies
        history.append(
            {
                "role": "assistant",
                "content": with_notice(
                    f"**Unexpected error.**\n\n`{type(exc).__name__}: {exc}`"
                ),
            }
        )
        return history, "", IDLE_METRICS

    reviewed, _flagged = safety.review_answer(answer)
    reviewed = with_notice(reviewed)

    history.append({"role": "assistant", "content": reviewed})
    return history, "", render_live_metrics(metrics)


def on_use_case_change(use_case_label: str) -> str:
    try:
        use_case = prompts.get_use_case(use_case_label)
    except KeyError:
        return ""
    return f"{use_case.description}\n\nExample: *{use_case.example}*"


def on_unload_click() -> str:
    RUNTIME.unload()
    return IDLE_METRICS + "\n\n_Model unloaded; GPU memory released._"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_interface() -> gr.Blocks:
    default_model = model_registry.default_model()
    default_use_case = prompts.USE_CASES_BY_KEY[prompts.DEFAULT_USE_CASE_KEY]

    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER_MARKDOWN)

        with gr.Tabs():
            # ---------------------------------------------------------- #
            with gr.Tab("Health Information Assistant"):
                # Controls sit in a left rail beside the conversation so the model
                # and use case stay visible and switchable while chatting.
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        model_dropdown = gr.Dropdown(
                            choices=model_registry.choice_labels(),
                            value=default_model.label,
                            label="Model",
                            info="Quantized checkpoints from the benchmark notebook.",
                        )
                        use_case_dropdown = gr.Dropdown(
                            choices=prompts.use_case_labels(),
                            value=default_use_case.label,
                            label="Use Case",
                        )
                        use_case_hint = gr.Markdown(
                            on_use_case_change(default_use_case.label)
                        )
                        metrics_panel = gr.Markdown(IDLE_METRICS)
                        with gr.Accordion("Selected model details", open=False):
                            model_card = gr.Markdown(render_model_card(default_model.label))
                        unload_button = gr.Button("Unload model / free GPU memory", size="sm")
                        with gr.Accordion("Scope of this demo", open=False):
                            gr.Markdown(SCOPE_MARKDOWN)

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            type="messages",
                            height=520,
                            show_copy_button=True,
                            allow_tags=False,
                            placeholder=(
                                "### Health Information Assistant\n\n"
                                "Ask a general health-information question, or pick "
                                "one of the examples below.\n\n"
                                "This demo explains and informs. It does not diagnose."
                            ),
                        )
                        with gr.Row():
                            question_box = gr.Textbox(
                                placeholder="Ask a health-information question...",
                                label="Your question",
                                scale=8,
                                autofocus=True,
                            )
                            send_button = gr.Button("Send", variant="primary", scale=1)
                        with gr.Row():
                            clear_button = gr.Button("Clear conversation", size="sm")

                        gr.Examples(
                            examples=[[use_case.example] for use_case in prompts.USE_CASES],
                            inputs=question_box,
                            label="Example questions",
                        )
                        gr.Markdown(
                            "_Each answer is generated from the current question "
                            "alone. The conversation stays visible, but earlier turns "
                            "are not fed back into the model: multi-turn behaviour was "
                            "not part of the benchmark and has not been evaluated here._"
                        )

                chat_inputs = [question_box, chatbot, model_dropdown, use_case_dropdown]
                chat_outputs = [chatbot, question_box, metrics_panel]
                send_button.click(respond, chat_inputs, chat_outputs)
                question_box.submit(respond, chat_inputs, chat_outputs)
                clear_button.click(lambda: ([], "", IDLE_METRICS), None, chat_outputs)
                use_case_dropdown.change(
                    on_use_case_change, use_case_dropdown, use_case_hint
                )
                model_dropdown.change(render_model_card, model_dropdown, model_card)
                unload_button.click(on_unload_click, None, metrics_panel)

            # ---------------------------------------------------------- #
            with gr.Tab("Benchmark Explorer"):
                problems = br.problems_markdown(RESULTS)
                if problems:
                    gr.Markdown(problems)

                gr.Markdown(br.maturity_banner(RESULTS))

                if RESULTS.available:
                    gr.Markdown("## Accuracy")
                    gr.Markdown(
                        "Primary metric: constrained scoring (medical knowledge "
                        "accuracy). Free-generation format results are a separate "
                        "secondary metric in the notebook and are not shown as "
                        "accuracy here."
                    )
                    gr.Dataframe(
                        value=RESULTS.table("accuracy"), interactive=False, wrap=True
                    )

                    gr.Markdown("## Performance")
                    gr.Dataframe(
                        value=RESULTS.table("performance"), interactive=False, wrap=True
                    )

                    gr.Markdown("## Memory")
                    gr.Dataframe(
                        value=RESULTS.table("memory"), interactive=False, wrap=True
                    )

                    gr.Markdown("## Quantization Trade-off")
                    gr.Markdown(
                        "Each quantized configuration is compared against the "
                        "baseline of its **own family**. `Latency Speedup` below 1.0 "
                        "means the quantized configuration was slower in this run."
                    )
                    gr.Dataframe(
                        value=RESULTS.table("tradeoff"), interactive=False, wrap=True
                    )

                gr.Markdown(br.environment_markdown(RESULTS))
                gr.Markdown(render_demo_runtime())

                provenance = br.provenance_markdown(RESULTS)
                if provenance:
                    with gr.Accordion("Result provenance", open=False):
                        gr.Markdown(provenance)

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
