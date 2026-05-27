"""Tests for OpenAI-compatible API server endpoints."""

from fastapi.testclient import TestClient
from src.api import server


class DummyEngine:
    """Simple deterministic engine for API tests."""

    def generate(self, params):
        return f"gen::{params.prompt}::{params.max_tokens}::{params.temperature}::{params.top_p}"

    def generate_batch(self, params_list):
        return [self.generate(params) for params in params_list]


def test_completions_handles_prompt_batch() -> None:
    """Ensure /v1/completions returns one choice per prompt in batch order."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post(
        "/v1/completions",
        json={"prompt": ["first", "second"], "max_tokens": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [choice["index"] for choice in payload["choices"]] == [0, 1]
    assert payload["choices"][0]["text"] == "gen::first::3::1.0::1.0"
    assert payload["choices"][1]["text"] == "gen::second::3::1.0::1.0"


def test_completions_passes_sampling_params() -> None:
    """Ensure completion endpoint forwards temperature and top_p to engine."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post(
        "/v1/completions",
        json={"prompt": "first", "max_tokens": 2, "temperature": 0.3, "top_p": 0.8},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "gen::first::2::0.3::0.8"


def test_completions_rejects_empty_prompt_batch() -> None:
    """Ensure /v1/completions returns 400 for empty prompt list."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post("/v1/completions", json={"prompt": []})

    assert response.status_code == 400
    assert "Prompt list cannot be empty" in response.json()["detail"]




def test_completions_rejects_blank_prompts() -> None:
    """Ensure /v1/completions returns 400 for blank prompt values."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post("/v1/completions", json={"prompt": "   "})

    assert response.status_code == 400
    assert "Prompt must contain non-whitespace characters" in response.json()["detail"]


def test_completions_forwards_turboquant_flag() -> None:
    """Ensure /v1/completions forwards turboquant_kv flag to engine params."""

    class CapturingEngine(DummyEngine):
        def __init__(self) -> None:
            self.seen_turboquant: list[bool] = []

        def generate_batch(self, params_list):
            self.seen_turboquant = [bool(params.turboquant_kv) for params in params_list]
            return super().generate_batch(params_list)

    engine = CapturingEngine()
    server._engine = engine
    client = TestClient(server.app)

    response = client.post(
        "/v1/completions",
        json={"prompt": "hello", "max_tokens": 2, "turboquant_kv": True},
    )

    assert response.status_code == 200
    assert engine.seen_turboquant == [True]

def test_chat_completions_uses_chat_template_markers() -> None:
    """Ensure chat endpoint builds prompt with message boundary markers."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "hello"},
            ]
        },
    )

    assert response.status_code == 200
    generated = response.json()["choices"][0]["message"]["content"]
    assert "<|im_start|>system" in generated
    assert "<|im_start|>user" in generated
    assert "<|im_start|>assistant" in generated




def test_completions_returns_nonzero_usage_counts() -> None:
    """Ensure /v1/completions usage contains derived token counts."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post(
        "/v1/completions",
        json={"prompt": "alpha beta", "max_tokens": 2},
    )

    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_returns_nonzero_usage_counts() -> None:
    """Ensure /v1/chat/completions usage contains derived token counts."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello world"}]},
    )

    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

def test_chat_completions_rejects_stream_mode_until_true_streaming_exists() -> None:
    """Ensure stream mode fails explicitly instead of emulating SSE with one full chunk."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert response.status_code == 501
    assert "stream=True is not supported yet" in response.json()["detail"]
