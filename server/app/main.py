import uuid
from fastapi import FastAPI, Depends, Request
from .config import DEFAULT_MODEL, OLLAMA_BASE_URL, RATE_LIMIT_PER_MINUTE, MAX_CONTEXT_TOKENS
from .auth import verify_api_key
from .rate_limit import RateLimiter
from .models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, Usage, HealthResponse
from .ollama_client import OllamaClient

app = FastAPI(title="Private LLM Service", version="1.0.0")
ollama = OllamaClient()
limiter = RateLimiter(max_requests=RATE_LIMIT_PER_MINUTE)


@app.get("/health", response_model=HealthResponse)
async def health():
    ok = await ollama.health()
    return HealthResponse(
        status="ok" if ok else "ollama_unavailable",
        model=DEFAULT_MODEL,
        ollama_url=OLLAMA_BASE_URL,
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
):
    limiter.check(request)

    model = body.model or DEFAULT_MODEL
    messages = [m.model_dump() for m in body.messages]

    # Enforce max context: rough estimate 4 chars ≈ 1 token
    total_chars = sum(len(m["content"]) for m in messages)
    approx_tokens = total_chars // 4
    if approx_tokens > MAX_CONTEXT_TOKENS:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Input too long: ~{approx_tokens} tokens, max {MAX_CONTEXT_TOKENS}",
        )

    result = await ollama.chat(
        model=model,
        messages=messages,
        temperature=body.temperature or 0.7,
        max_tokens=body.max_tokens,
    )

    reply = result.get("message", {}).get("content", "")
    prompt_tokens = result.get("prompt_eval_count", approx_tokens)
    completion_tokens = result.get("eval_count", len(reply) // 4)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
