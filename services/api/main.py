"""NeuroForge API.

The frontend is a static export and can talk to model providers directly with a
user-supplied key. This service exists for the deployment where you would rather
not put a key in a browser: it holds the credentials server-side and exposes a
single narrow endpoint that streams a circuit plan back to the client.

It is deliberately small. No database, no sessions, no user accounts — the
document lives in the browser's IndexedDB and never comes here except as context
for a planning request.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Deque, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_ANTHROPIC_MODEL = "claude-fable-5"
DEFAULT_OPENAI_MODEL = "gpt-5"

# A planning request carries the whole circuit as context. Cap it so a runaway
# client cannot push a 50 MB document through the upstream provider.
MAX_CIRCUIT_BYTES = 512 * 1024
MAX_PROMPT_CHARS = 8_000

RATE_LIMIT_REQUESTS = int(os.getenv("NEUROFORGE_RATE_LIMIT", "20"))
RATE_LIMIT_WINDOW_S = 60.0

_client: httpx.AsyncClient | None = None
_hits: dict[str, Deque[float]] = defaultdict(deque)
_rate_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    try:
        yield
    finally:
        await _client.aclose()
        _client = None


app = FastAPI(
    title="NeuroForge API",
    version="0.1.0",
    description="Circuit planning proxy for the NeuroForge editor.",
    lifespan=lifespan,
)

# The static frontend is served from a different origin (GitHub Pages), so CORS
# is mandatory rather than optional. Configure the exact origins in deployment;
# the default is permissive only because there is nothing here worth stealing.
_origins = os.getenv("NEUROFORGE_ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str | None = None
    # The circuit is passed through verbatim as JSON context. It is not validated
    # against the document schema here; the client already did that, and this
    # service has no reason to know the schema.
    circuit: dict[str, Any] = Field(default_factory=dict)
    system: str | None = Field(default=None, max_length=32_000)
    tool_schema: dict[str, Any] | None = None

    @field_validator("circuit")
    @classmethod
    def _bounded(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value)) > MAX_CIRCUIT_BYTES:
            raise ValueError("circuit context exceeds the size limit")
        return value


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _enforce_rate_limit(request: Request) -> None:
    key = _client_key(request)
    now = time.monotonic()
    async with _rate_lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_S:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW_S - (now - bucket[0])) + 1
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
        # Keep the table from growing without bound under churning client IPs.
        if len(_hits) > 4096:
            for stale in [k for k, v in _hits.items() if not v or now - v[-1] > 300]:
                del _hits[stale]


def _resolve_key(provider: str, override: str | None) -> str:
    if override:
        return override
    env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    key = os.getenv(env)
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{env} is not configured on this server, and no key was supplied "
                "in the X-Provider-Key header"
            ),
        )
    return key


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "providers": {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
        },
        "rateLimit": {"requests": RATE_LIMIT_REQUESTS, "windowSeconds": RATE_LIMIT_WINDOW_S},
    }


def _anthropic_payload(body: PlanRequest) -> dict[str, Any]:
    user_content = (
        f"{body.prompt}\n\n<current_circuit>\n"
        f"{json.dumps(body.circuit, separators=(',', ':'))}\n</current_circuit>"
    )
    payload: dict[str, Any] = {
        "model": body.model or DEFAULT_ANTHROPIC_MODEL,
        "max_tokens": 8192,
        "stream": True,
        "messages": [{"role": "user", "content": user_content}],
    }
    if body.system:
        payload["system"] = body.system
    if body.tool_schema:
        payload["tools"] = [body.tool_schema]
        payload["tool_choice"] = {"type": "tool", "name": body.tool_schema.get("name", "build_circuit")}
    return payload


def _openai_payload(body: PlanRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{body.prompt}\n\n<current_circuit>\n"
                f"{json.dumps(body.circuit, separators=(',', ':'))}\n</current_circuit>"
            ),
        }
    )
    payload: dict[str, Any] = {
        "model": body.model or DEFAULT_OPENAI_MODEL,
        "messages": messages,
        "stream": True,
    }
    if body.tool_schema:
        payload["tools"] = [{"type": "function", "function": body.tool_schema}]
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": body.tool_schema.get("name", "build_circuit")},
        }
    return payload


@app.post("/v1/plan")
async def plan(
    body: PlanRequest,
    request: Request,
    x_provider_key: str | None = Header(default=None, alias="X-Provider-Key"),
) -> StreamingResponse:
    """Stream a circuit plan from the upstream provider.

    The upstream server-sent-event stream is relayed to the client unmodified.
    Parsing is the client's job — it already owns the plan schema, and reshaping
    the events here would mean this service had to track two provider formats
    plus its own.
    """
    await _enforce_rate_limit(request)

    if _client is None:
        raise HTTPException(status_code=503, detail="http client unavailable")

    api_key = _resolve_key(body.provider, x_provider_key)

    if body.provider == "anthropic":
        url = ANTHROPIC_URL
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = _anthropic_payload(body)
    else:
        url = OPENAI_URL
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        payload = _openai_payload(body)

    async def relay() -> AsyncIterator[bytes]:
        assert _client is not None
        try:
            async with _client.stream("POST", url, headers=headers, json=payload) as upstream:
                if upstream.status_code >= 400:
                    detail = (await upstream.aread()).decode("utf-8", "replace")[:2000]
                    yield _sse_error(f"upstream {upstream.status_code}: {detail}")
                    return
                async for chunk in upstream.aiter_raw():
                    yield chunk
        except httpx.TimeoutException:
            yield _sse_error("upstream request timed out")
        except httpx.HTTPError as exc:
            yield _sse_error(f"upstream transport error: {exc}")

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse_error(message: str) -> bytes:
    return f"event: error\ndata: {json.dumps({'error': message})}\n\n".encode()


@app.exception_handler(HTTPException)
async def _http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=exc.headers or {},
    )
