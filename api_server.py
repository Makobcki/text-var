"""OpenAI-compatible API server for TextVAR model."""
#!/usr/bit/env python

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
from pipeline import PipelineConfig, TextVARPipeline
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# Глобальный инстанс модели (чтобы загружать веса 1 раз)
_pipeline: TextVARPipeline | None = None


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
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet.")

    # Простой конвертер истории чата в текст (пока без сложных chat templates)
    prompt = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in request.messages])
    prompt += "\nAssistant: "

    try:
        generated_text = _pipeline.generate(prompt, max_new_tokens=request.max_tokens)
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
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet.")

    prompt_str = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    try:
        generated_text = _pipeline.generate(prompt_str, max_new_tokens=request.max_tokens)
    except Exception as e:
        LOGGER.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(text=generated_text, index=0)],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


# --- Entrypoint ---


def main() -> None:
    global _pipeline

    parser = argparse.ArgumentParser(description="VAR OpenAI-Compatible API Server")
    parser.add_argument("--vqvae-path", type=Path, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--var-path", type=Path, required=True, help="Path to VAR checkpoint")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to BPE tokenizer JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--max-bpe-len", type=int, default=128)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action="store_true")
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
    LOGGER.info("Models loaded successfully.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
