"""Tests for OpenAI-compatible API server endpoints."""

from fastapi.testclient import TestClient

from src.api import server


class DummyEngine:
    """Simple deterministic engine for API tests."""

    def generate(self, params):
        return f"gen::{params.prompt}::{params.max_tokens}"

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
    assert payload["choices"][0]["text"] == "gen::first::3"
    assert payload["choices"][1]["text"] == "gen::second::3"


def test_completions_rejects_empty_prompt_batch() -> None:
    """Ensure /v1/completions returns 400 for empty prompt list."""
    server._engine = DummyEngine()
    client = TestClient(server.app)

    response = client.post("/v1/completions", json={"prompt": []})

    assert response.status_code == 400
    assert "Prompt list cannot be empty" in response.json()["detail"]


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
