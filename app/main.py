from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import chat_v2, chat, latency
from app.bootstrap import bootstrap_services

app = FastAPI(
    title="ICD-11 Enhanced RAG API - Intelligent Fusion",
    description="Enhanced API with Intelligent RAG and Fusion systems for comprehensive mental health assistance",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    bootstrap_services()

app.include_router(chat_v2.router)
app.include_router(chat.router)
app.include_router(latency.router)

@app.get("/")
async def root():
    return {
        "message": "ICD-11 Enhanced RAG API with CBT Integration",
        "version": "4.0.0",
        "endpoints": {
            "chat_v2": "/api/v2/empathetic_professional",
            "chat": "/api/empathetic_professional",
            "latency_ping": "/api/latency/ping",
            "latency_quick_reply": "/api/latency/quick_reply",
        },
    }


