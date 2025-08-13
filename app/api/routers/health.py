from fastapi import APIRouter
from app.schemas.chat import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check_stub():
    return HealthResponse(
        status="healthy",
        psychologist_llm_loaded=False,
        store_loaded=False,
        device="cpu",
        gpu_memory=None,
        opro_prompt_loaded=False,
        interactions_count=0,
        cbt_available=False,
        cbt_techniques=0,
        cbt_content=0,
        cbt_smoke_test_passed=False,
        enhanced_systems_available=False,
        enhanced_rag_loaded=False,
        intelligent_fusion_loaded=False,
        resolved_paths={},
    )


