# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import chat_v2, chat, latency
from app.bootstrap import bootstrap_services

# --- E/S/Q global defaults (routers can read from app.state.*) ---
ESQ_STOP = ["<END>"]
ESQ_MAX_NEW_TOKENS = 90
ESQ_TEMPERATURE = 0.3
ESQ_TOP_P = 0.8
ESQ_WORD_LIMIT = 120

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    bootstrap_services()
    # expose decoding/format settings to routers
    app.state.esq_stop = ESQ_STOP
    app.state.esq_max_new_tokens = ESQ_MAX_NEW_TOKENS
    app.state.esq_temperature = ESQ_TEMPERATURE
    app.state.esq_top_p = ESQ_TOP_P
    app.state.esq_word_limit = ESQ_WORD_LIMIT
    yield
    # Shutdown (add any clean-up here if needed)

app = FastAPI(
    title="ICD-11 Enhanced RAG API - Intelligent Fusion",
    description="Enhanced API with Intelligent RAG and Fusion systems for comprehensive mental health assistance",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global error guards so frontend will never see an empty payload ---
@app.exception_handler(Exception)
async def _unhandled_exc(_: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)},
    )

# Routers
app.include_router(chat_v2.router)
app.include_router(chat.router)
app.include_router(latency.router)

# Health/Info
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": "4.0.0"}

@app.get("/")
async def root():
    return {
        "message": "ICD-11 Enhanced RAG API with CBT Integration",
        "version": "4.0.0",
        "esq": {
            "stop": ESQ_STOP,
            "max_new_tokens": ESQ_MAX_NEW_TOKENS,
            "temperature": ESQ_TEMPERATURE,
            "top_p": ESQ_TOP_P,
            "word_limit": ESQ_WORD_LIMIT,
        },
        "endpoints": {
            "chat_v2": "/api/v2/empathetic_professional",
            "chat": "/api/empathetic_professional",
            "latency_ping": "/api/latency/ping",
            "latency_quick_reply": "/api/latency/quick_reply",
        },
    }
