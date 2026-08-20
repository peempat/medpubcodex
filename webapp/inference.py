"""Model runtime for the Health Information Assistant demo.

One model is resident at a time. Selecting a different model unloads the current
one, clears the CUDA allocator, then loads the new one; selecting the same model
again reuses what is already loaded.

Loading mirrors the benchmark notebook: ``AutoProcessor`` plus
``AutoModelForMultimodalLM``, with the processor wrapped in a text-only adapter.
The quantized entries are loaded straight from the BitsAndBytes checkpoints the
notebook pushed, so no quantization config is rebuilt here.

``torch`` and ``transformers`` are imported lazily so that the Benchmark Explorer
tab still works on a machine with no CUDA build installed.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Any

from model_registry import ModelEntry

# Conservative generation settings for a demo. Greedy decoding keeps answers
# reproducible; the benchmark used greedy decoding as well.
MAX_NEW_TOKENS = 320
DO_SAMPLE = False

TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


class InferenceError(RuntimeError):
    """An error that carries a message safe to show in the UI."""


class MissingTokenError(InferenceError):
    pass


class CudaUnavailableError(InferenceError):
    pass


class OutOfMemoryError(InferenceError):
    pass


class ModelLoadError(InferenceError):
    pass


class GenerationError(InferenceError):
    pass


@dataclass
class GenerationMetrics:
    """Metrics measured during one live demo request.

    These are runtime numbers from this machine and this request. They are not
    comparable with the controlled benchmark numbers in the Benchmark Explorer.
    """

    model_label: str
    precision: str
    repo_id: str
    latency_seconds: float
    generated_tokens: int
    tokens_per_second: float
    prompt_tokens: int
    peak_vram_gib: float | None
    device: str
    load_time_seconds: float | None = None
    was_freshly_loaded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def get_hf_token() -> str | None:
    """Return the Hugging Face token from the environment, if any."""
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise InferenceError(
            "PyTorch is not installed. Install the app requirements first: "
            "pip install -r webapp/requirements.txt"
        ) from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise InferenceError(
            "transformers is not installed, or it is older than the version this "
            "project uses. The benchmark ran on transformers 5.14.1, which "
            "provides AutoModelForMultimodalLM."
        ) from exc
    return AutoModelForMultimodalLM, AutoProcessor


def describe_runtime() -> dict[str, Any]:
    """Describe the machine this demo is running on.

    Kept separate from the benchmark environment recorded in the results
    manifest: the demo may run on completely different hardware.
    """
    import platform
    import sys

    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "torch_version": None,
        "transformers_version": None,
        "bitsandbytes_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_total_memory_gib": None,
        "cpu_ram_gib": None,
        "hf_token_present": get_hf_token() is not None,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_total_memory_gib"] = round(props.total_memory / 2**30, 2)
    except Exception:
        pass

    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:
        pass

    try:
        import bitsandbytes

        info["bitsandbytes_version"] = getattr(bitsandbytes, "__version__", "unknown")
    except Exception:
        pass

    try:
        import psutil

        info["cpu_ram_gib"] = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        pass

    del sys
    return info


class TextOnlyProcessor:
    """Tokenizer-shaped view over a multimodal processor.

    Mirrors the adapter the benchmark notebook uses so that prompt rendering in
    the demo matches how the models were evaluated.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.chat_template = getattr(processor, "chat_template", None) or getattr(
            self.tokenizer, "chat_template", None
        )

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def __call__(self, text, return_tensors="pt", **kwargs):
        return self.processor(text=text, return_tensors=return_tensors, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        multimodal = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            multimodal.append({"role": message["role"], "content": content})
        return self.processor.apply_chat_template(
            multimodal,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Fold a system message into the first user turn.

    Some Gemma chat templates reject a ``system`` role. The instruction still has
    to reach the model, so it is prefixed to the user turn instead.
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    if not system or not rest:
        return rest or messages
    merged = dict(rest[0])
    merged["content"] = f"{system}\n\n---\n\n{merged['content']}"
    return [merged] + rest[1:]


class ModelRuntime:
    """Holds at most one loaded model and serves single-turn generations."""

    def __init__(self) -> None:
        self._entry: ModelEntry | None = None
        self._model = None
        self._processor: TextOnlyProcessor | None = None
        self._load_time: float | None = None

    # ------------------------------------------------------------------ #
    @property
    def loaded_entry(self) -> ModelEntry | None:
        return self._entry

    def status(self) -> str:
        if self._entry is None:
            return "No model loaded."
        return f"Loaded: {self._entry.label} ({self._entry.repo_id})"

    # ------------------------------------------------------------------ #
    def unload(self) -> None:
        """Drop the resident model and release its GPU memory."""
        self._model = None
        self._processor = None
        self._entry = None
        self._load_time = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def ensure_loaded(self, entry: ModelEntry) -> bool:
        """Load ``entry`` if it is not already resident.

        Returns:
            True if a load actually happened.

        Raises:
            InferenceError: with a message suitable for display.
        """
        if self._entry is not None and self._entry.key == entry.key and self._model is not None:
            return False

        torch = _import_torch()
        token = get_hf_token()

        if entry.requires_cuda and not torch.cuda.is_available():
            reason = (
                "is a BitsAndBytes checkpoint, and BitsAndBytes needs a CUDA GPU"
                if entry.loads_prequantized
                else "is an unquantized multimodal model that is not practical to serve on CPU"
            )
            raise CudaUnavailableError(
                f"{entry.label} {reason}, but PyTorch reports no CUDA device on this "
                "machine.\n\n"
                f"Detected PyTorch build: {torch.__version__}\n\n"
                "Run the demo on a CUDA machine, or open the Benchmark Explorer tab, "
                "which needs no GPU."
            )

        if entry.is_private and token is None:
            raise MissingTokenError(
                f"{entry.label} lives in the private repository `{entry.repo_id}`.\n\n"
                "Set a Hugging Face access token with read permission for that "
                "repository, then restart the app:\n\n"
                "- PowerShell: `$env:HF_TOKEN = \"hf_...\"`\n"
                "- Linux/macOS: `export HF_TOKEN=hf_...`\n"
                "- Or copy `webapp/.env.example` to `webapp/.env` and fill it in."
            )

        # Free the previous model before allocating the next one.
        self.unload()

        AutoModelForMultimodalLM, AutoProcessor = _import_transformers()

        common: dict[str, Any] = {"trust_remote_code": False}
        if entry.revision:
            common["revision"] = entry.revision
        if token:
            common["token"] = token

        started = time.perf_counter()
        try:
            processor = AutoProcessor.from_pretrained(entry.repo_id, **common)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

            model_kwargs: dict[str, Any] = dict(common)
            model_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "cpu"
            if not entry.loads_prequantized:
                # Baseline: match the notebook dtype choice.
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    model_kwargs["dtype"] = torch.bfloat16
                elif torch.cuda.is_available():
                    model_kwargs["dtype"] = torch.float16

            model = AutoModelForMultimodalLM.from_pretrained(entry.repo_id, **model_kwargs).eval()
        except Exception as exc:
            self.unload()
            raise self._translate_load_error(exc, entry) from exc

        self._model = model
        self._processor = TextOnlyProcessor(processor)
        self._entry = entry
        self._load_time = time.perf_counter() - started
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def _translate_load_error(exc: Exception, entry: ModelEntry) -> InferenceError:
        """Turn a loader exception into a message that helps the user."""
        text = f"{type(exc).__name__}: {exc}"
        lowered = text.lower()

        if "out of memory" in lowered or "cuda error" in lowered and "memory" in lowered:
            return OutOfMemoryError(
                f"The GPU ran out of memory while loading {entry.label}.\n\n"
                "Pick a smaller configuration (the INT4 entries need the least "
                "VRAM), or free GPU memory and try again."
            )
        if "401" in text or "unauthorized" in lowered or "authentication" in lowered:
            return MissingTokenError(
                f"Hugging Face rejected the credentials for `{entry.repo_id}`.\n\n"
                "Check that HF_TOKEN is set and that the token is still valid."
            )
        if "403" in text or "gated" in lowered or "awaiting a review" in lowered:
            return ModelLoadError(
                f"Access to `{entry.repo_id}` was refused.\n\n"
                "The Gemma and MedGemma repositories are gated, and the quantized "
                "copies are private. Accept the model licence on Hugging Face and "
                "make sure your token has read access to that repository."
            )
        if "404" in text or "not found" in lowered or "repositorynotfound" in lowered:
            return ModelLoadError(
                f"Repository `{entry.repo_id}` was not found.\n\n"
                "It may have been renamed or removed. The quantized repositories "
                "were created by the benchmark notebook upload cell."
            )
        if "bitsandbytes" in lowered:
            return ModelLoadError(
                f"{entry.label} needs the bitsandbytes runtime, which failed to load.\n\n"
                f"Details: {text}\n\n"
                "bitsandbytes requires a CUDA build of PyTorch."
            )
        if "connection" in lowered or "timeout" in lowered or "offline" in lowered:
            return ModelLoadError(
                f"Could not reach Hugging Face while loading `{entry.repo_id}`.\n\n"
                f"Details: {text}"
            )
        return ModelLoadError(f"Could not load {entry.label}.\n\nDetails: {text}")

    # ------------------------------------------------------------------ #
    def _render_prompt(self, messages: list[dict]) -> str:
        assert self._processor is not None
        if not self._processor.chat_template:
            raise GenerationError(
                "The selected model has no chat template. These instruction-tuned "
                "models are evaluated with their own chat template, so the demo "
                "cannot fall back to a raw prompt."
            )
        try:
            return self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Several Gemma templates reject a standalone system role.
            return self._processor.apply_chat_template(
                _merge_system_into_user(messages), tokenize=False, add_generation_prompt=True
            )

    def generate(self, messages: list[dict], entry: ModelEntry) -> tuple[str, GenerationMetrics]:
        """Run one single-turn generation and measure it.

        Args:
            messages: Output of :func:`prompts.build_messages`.
            entry: The model configuration to use.

        Returns:
            ``(answer_text, metrics)``.

        Raises:
            InferenceError: with a message suitable for display.
        """
        torch = _import_torch()
        freshly_loaded = self.ensure_loaded(entry)
        assert self._model is not None and self._processor is not None

        prompt = self._render_prompt(messages)
        device = str(getattr(self._model, "device", "cpu"))

        try:
            inputs = self._processor(prompt, return_tensors="pt")
            inputs = {
                key: (value.to(self._model.device) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            prompt_tokens = int(inputs["input_ids"].shape[-1])

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            started = time.perf_counter()
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    use_cache=True,
                    pad_token_id=self._processor.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - started

            new_tokens = output[0][prompt_tokens:]
            generated_tokens = int(new_tokens.shape[-1])
            answer = self._processor.decode(new_tokens, skip_special_tokens=True).strip()

            peak_vram = (
                round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if torch.cuda.is_available()
                else None
            )
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if "out of memory" in text.lower():
                self.unload()
                raise OutOfMemoryError(
                    "The GPU ran out of memory during generation.\n\n"
                    "The model was unloaded. Try a smaller configuration (INT4 uses "
                    "the least VRAM) or a shorter question."
                ) from exc
            raise GenerationError(f"Generation failed.\n\nDetails: {text}") from exc

        metrics = GenerationMetrics(
            model_label=entry.label,
            precision=entry.precision,
            repo_id=entry.repo_id,
            latency_seconds=round(latency, 3),
            generated_tokens=generated_tokens,
            tokens_per_second=round(generated_tokens / latency, 2) if latency > 0 else 0.0,
            prompt_tokens=prompt_tokens,
            peak_vram_gib=peak_vram,
            device=device,
            load_time_seconds=round(self._load_time, 2) if self._load_time else None,
            was_freshly_loaded=freshly_loaded,
        )

        if not answer:
            answer = (
                "The model returned an empty answer. Try rephrasing the question, "
                "or select a different model configuration."
            )
        return answer, metrics


RUNTIME = ModelRuntime()
