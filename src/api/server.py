"""OpenAI-compatible API server for TextVAR model."""
#!/usr/bin/env python

import argparse
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from src.api.cli import build_parser
from src.api.engine import GenerationParams, TextVAREngine
from src.core.pipeline import PipelineConfig, TextVARPipeline
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# Глобальный инстанс модели (чтобы загружать веса 1 раз)
_pipeline: TextVARPipeline | None = None
_engine: TextVAREngine | None = None


# --- Pydantic Models for OpenAI API Contract ---


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "var-team"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "text-var"
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "text-var"
    prompt: str | list[str]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    """Build a ChatML-like prompt from chat messages.

    Args:
        messages: Chronological chat messages.

    Returns:
        Prompt string ready for model generation.
    """
    prompt_parts = [f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>" for msg in messages]
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def _normalize_prompts(prompt: str | list[str]) -> list[str]:
    """Normalize completion prompts to a non-empty list.

    Args:
        prompt: Either single prompt string or prompt batch.

    Returns:
        A list of prompts preserving input order.

    Raises:
        ValueError: If the prompt batch is empty.
    """
    if isinstance(prompt, str):
        return [prompt]
    if not prompt:
        raise ValueError("Prompt list cannot be empty.")
    return prompt


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict[str, int]


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: dict[str, int]


# --- FastAPI Lifecycle & Endpoints ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan hook to ensure model is loaded before serving requests."""
    if _pipeline is None:
        LOGGER.warning("Pipeline is not initialized! Requests will fail.")
    else:
        LOGGER.info("API is ready to serve requests.")
    yield


app = FastAPI(title="TextVAR OpenAI API", version="1.0.0", lifespan=lifespan)


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List available models (Mocked for OpenAI compatibility)."""
    return ModelList(data=[ModelCard(id="text-var")])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat response (transforms messages to flat prompt)."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet.")

    prompt = _build_chat_prompt(request.messages)

    try:
        generated_text = _engine.generate(GenerationParams(prompt=prompt, max_tokens=request.max_tokens))
    except Exception as e:
        LOGGER.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(index=0, message=ChatMessage(role="assistant", content=generated_text))
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},  # Mocked usage
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest) -> CompletionResponse:
    """Generate basic text completion."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet.")

    try:
        prompts = _normalize_prompts(request.prompt)
        generated_texts = [
            _engine.generate(GenerationParams(prompt=prompt, max_tokens=request.max_tokens))
            for prompt in prompts
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        LOGGER.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(text=generated_text, index=index)
            for index, generated_text in enumerate(generated_texts)
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


# --- Entrypoint ---


def main() -> None:
    global _pipeline, _engine

    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    LOGGER.info("Loading models... This may take a moment.")
    cfg = PipelineConfig(
        vqvae_path=args.vqvae_path,
        var_path=args.var_path,
        bpe_tokenizer_path=args.tokenizer,
        device=args.device,
        max_bpe_len=args.max_bpe_len,
    )
    _pipeline = TextVARPipeline(cfg)
    _engine = TextVAREngine(_pipeline)
    LOGGER.info("Models loaded successfully.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
