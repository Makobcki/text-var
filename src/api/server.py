"""OpenAI-compatible API server for TextVAR model."""
#!/usr/bin/env python

import asyncio
import logging
import uuid
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from src.api.cli import build_parser
from src.api.engine import GenerationParams, TextVAREngine
from src.core.pipeline import PipelineConfig, TextVARPipeline
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# Глобальный инстанс модели (чтобы загружать веса 1 раз)
_pipeline: TextVARPipeline | None = None
_engine: TextVAREngine | None = None


class _BatchTask:
    """Internal asynchronous completion task for continuous batching."""

    def __init__(self, params: GenerationParams) -> None:
        self.params = params
        self.future: asyncio.Future[str] = asyncio.get_running_loop().create_future()


class ContinuousBatchingProcessor:
    """Continuously merges pending generation tasks into dynamic batches."""

    def __init__(self, max_batch_size: int = 16, collect_window_ms: int = 8) -> None:
        self._max_batch_size = max_batch_size
        self._collect_window_seconds = collect_window_ms / 1000
        self._queue: asyncio.Queue[_BatchTask] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background batch worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        """Stop the background batch worker."""
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            LOGGER.debug("Continuous batching worker cancelled.")
        self._worker_task = None

    async def generate(self, params: GenerationParams) -> str:
        """Queue generation request and wait for result."""
        task = _BatchTask(params)
        await self._queue.put(task)
        return await task.future

    async def _run_worker(self) -> None:
        """Process queued requests with dynamic batch formation."""
        while True:
            first_task = await self._queue.get()
            pending: list[_BatchTask] = [first_task]

            deadline = asyncio.get_running_loop().time() + self._collect_window_seconds
            while len(pending) < self._max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            if _engine is None:
                for task in pending:
                    if not task.future.done():
                        task.future.set_exception(RuntimeError("Model is not loaded."))
                continue

            params_batch = [task.params for task in pending]
            try:
                outputs = await run_in_threadpool(_engine.generate_batch, params_batch)
            except Exception as exc:
                for task in pending:
                    if not task.future.done():
                        task.future.set_exception(exc)
                continue

            for output, task in zip(outputs, pending):
                if not task.future.done():
                    task.future.set_result(output)


_batch_processor = ContinuousBatchingProcessor()


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
    turboquant_kv: bool = False


class CompletionRequest(BaseModel):
    model: str = "text-var"
    prompt: str | list[str]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    turboquant_kv: bool = False


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
        normalized = [prompt]
    else:
        if not prompt:
            raise ValueError("Prompt list cannot be empty.")
        normalized = prompt

    if any(not item.strip() for item in normalized):
        raise ValueError("Prompt must contain non-whitespace characters.")
    return normalized


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
    await _batch_processor.start()
    yield
    await _batch_processor.stop()


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

    prompt = _build_chat_prompt(request.messages)

    params = GenerationParams(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        turboquant_kv=request.turboquant_kv,
    )
    try:
        generated_text = await run_in_threadpool(_engine.generate, params)
    except Exception as e:
        LOGGER.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="stream=True is not supported yet: token-level streaming generator is not implemented.",
        )

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

    try:
        prompts = _normalize_prompts(request.prompt)
        params_batch = [
            GenerationParams(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                turboquant_kv=request.turboquant_kv,
            )
            for prompt in prompts
        ]
        generated_texts = await asyncio.gather(
            *[_batch_processor.generate(params) for params in params_batch]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        LOGGER.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="stream=True is not supported yet: token-level streaming generator is not implemented.",
        )

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
