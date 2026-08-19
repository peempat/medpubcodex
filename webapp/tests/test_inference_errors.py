"""Tests for runtime error handling.

Real model loading is not exercised here: the checkpoints are private, gated,
and need a CUDA GPU. What is tested is that every failure the demo can hit turns
into a readable message instead of a traceback, and that no secret leaks into it.
"""

import os

import pytest

import inference
import model_registry

torch = pytest.importorskip("torch")


@pytest.fixture
def entry():
    return model_registry.default_model()


@pytest.fixture
def no_token(monkeypatch):
    for name in inference.TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend a CUDA device exists so later checks can be reached."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)


# --------------------------------------------------------------------------- #
def test_missing_cuda_is_reported_clearly(monkeypatch, entry):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(inference.CudaUnavailableError) as excinfo:
        inference.ModelRuntime().ensure_loaded(entry)
    message = str(excinfo.value)
    assert "no CUDA device" in message
    assert "Benchmark Explorer" in message


def test_missing_token_is_reported_before_any_download(fake_cuda, no_token, entry):
    with pytest.raises(inference.MissingTokenError) as excinfo:
        inference.ModelRuntime().ensure_loaded(entry)
    message = str(excinfo.value)
    assert entry.repo_id in message
    assert "HF_TOKEN" in message


def test_token_is_read_from_every_supported_variable(monkeypatch, no_token):
    assert inference.get_hf_token() is None
    for name in inference.TOKEN_ENV_VARS:
        monkeypatch.setenv(name, "hf_example")
        assert inference.get_hf_token() == "hf_example"
        monkeypatch.delenv(name)


def test_token_is_never_included_in_error_text(monkeypatch, fake_cuda, entry):
    """A load failure must not echo the secret back into the UI."""
    secret = "hf_thisisasecrettoken"
    monkeypatch.setenv("HF_TOKEN", secret)
    error = inference.ModelRuntime._translate_load_error(
        Exception(f"401 Client Error for token={secret}"), entry
    )
    assert secret not in str(error)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), inference.OutOfMemoryError),
        (Exception("401 Client Error: Unauthorized"), inference.MissingTokenError),
        (Exception("403 Client Error: gated repo"), inference.ModelLoadError),
        (Exception("404 RepositoryNotFoundError"), inference.ModelLoadError),
        (Exception("bitsandbytes was compiled without GPU support"), inference.ModelLoadError),
        (Exception("Connection timeout to huggingface.co"), inference.ModelLoadError),
        (Exception("something entirely unexpected"), inference.ModelLoadError),
    ],
)
def test_load_errors_are_translated(entry, raised, expected):
    error = inference.ModelRuntime._translate_load_error(raised, entry)
    assert isinstance(error, expected)
    assert isinstance(error, inference.InferenceError)
    assert str(error)


def test_out_of_memory_suggests_a_smaller_configuration(entry):
    error = inference.ModelRuntime._translate_load_error(
        RuntimeError("CUDA out of memory"), entry
    )
    assert "INT4" in str(error)


# --------------------------------------------------------------------------- #
def test_unload_is_safe_when_nothing_is_loaded():
    runtime = inference.ModelRuntime()
    runtime.unload()
    runtime.unload()
    assert runtime.loaded_entry is None
    assert runtime.status() == "No model loaded."


def test_runtime_description_never_raises():
    info = inference.describe_runtime()
    for key in ("python_version", "cuda_available", "hf_token_present", "torch_version"):
        assert key in info
    assert isinstance(info["cuda_available"], bool)


def test_runtime_description_reports_no_token_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
    info = inference.describe_runtime()
    assert info["hf_token_present"] is True
    assert "hf_supersecret" not in repr(info)


# --------------------------------------------------------------------------- #
def test_system_role_fallback_folds_into_the_user_turn():
    """Gemma chat templates that reject a system role must still get the instruction."""
    merged = inference._merge_system_into_user(
        [
            {"role": "system", "content": "SYSTEM RULES"},
            {"role": "user", "content": "What is hypertension?"},
        ]
    )
    assert len(merged) == 1
    assert merged[0]["role"] == "user"
    assert "SYSTEM RULES" in merged[0]["content"]
    assert "What is hypertension?" in merged[0]["content"]


def test_generation_settings_are_deterministic():
    assert inference.DO_SAMPLE is False, "the benchmark used greedy decoding"
    assert inference.MAX_NEW_TOKENS > 0
