import time
from typing import Optional
from fastapi import APIRouter, Response


router = APIRouter()


@router.get("/api/latency/ping")
async def latency_ping(response: Response):
    start = time.time()
    payload = {"message": "pong"}
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Response-Time"] = str(elapsed_ms)
    return {"elapsed_ms": elapsed_ms, **payload}


@router.post("/api/latency/quick_reply")
async def latency_quick_reply(
    response: Response,
    message: Optional[str] = None,
    repeat: int = 1,
    length: Optional[int] = None,
):
    start = time.time()
    base = message or (
        "I hear this is a lot to carry. Let's take one small step you can control today."
    )
    base = (base + " ") * max(1, repeat)
    if length and length > 0:
        if len(base) < length:
            base = (base + " ") * ((length // max(1, len(base))) + 1)
        base = base[:length]
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Response-Time"] = str(elapsed_ms)
    return {"answer": base, "length": len(base), "elapsed_ms": elapsed_ms}





