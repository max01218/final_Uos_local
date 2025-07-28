#!/usr/bin/env python3
"""
Enhanced FastAPI Integration with Intelligent Fusion System
Example integration of advanced RAG retrieval and CBT-ICD11 fusion
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import enhanced systems
try:
    from enhanced_rag_retriever import EnhancedRAGRetriever, RetrievalContext
    from intelligent_fusion_system import IntelligentFusionSystem, FusionContext, ProblemType
    ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced systems not available: {e}")
    ENHANCED_SYSTEMS_AVAILABLE = False

# Pydantic models for API
class EnhancedQueryRequest(BaseModel):
    message: str
    context: Optional[Dict] = None
    strategy: Optional[str] = "auto"
    urgency_level: Optional[int] = 1
    session_stage: Optional[str] = "initial"

class EnhancedQueryResponse(BaseModel):
    response: str
    confidence: float
    fusion_strategy: str
    source_breakdown: Dict[str, float]
    follow_up_suggestions: List[str]
    safety_notes: List[str]
    reasoning: str
    metadata: Dict

class SystemStatusResponse(BaseModel):
    enhanced_rag_available: bool
    fusion_system_available: bool
    knowledge_bases_loaded: Dict[str, bool]
    system_health: str
    last_updated: str

class RetrievalAnalysisRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    include_reasoning: Optional[bool] = True

class RetrievalAnalysisResponse(BaseModel):
    query: str
    intent_detected: str
    intent_confidence: float
    urgency_level: int
    retrieved_sources: List[Dict]
    total_results: int
    processing_time_ms: float

class EnhancedFastAPIIntegration:
    """Enhanced FastAPI integration with intelligent fusion capabilities"""
    
    def __init__(self, config_path: str = "intelligent_fusion_config.json"):
        self.app = FastAPI(
            title="Enhanced CBT-ICD11 Mental Health API",
            description="Advanced mental health support with intelligent fusion of CBT and ICD-11 knowledge",
            version="2.0.0"
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize enhanced systems
        self.enhanced_rag = None
        self.fusion_system = None
        self.system_ready = False
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_logging()
        
        # Initialize systems
        if ENHANCED_SYSTEMS_AVAILABLE:
            self._initialize_enhanced_systems()
            
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return self._default_config()
            
    def _default_config(self) -> Dict:
        """Default configuration if file loading fails"""
        return {
            "enhanced_rag_config": {
                "retrieval_weights": {
                    "semantic_weight": 0.5,
                    "keyword_weight": 0.3,
                    "context_weight": 0.2
                }
            },
            "fusion_system_config": {
                "response_generation": {
                    "max_response_length": 500
                }
            }
        }
        
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request timing middleware
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
    def _setup_logging(self):
        """Setup enhanced logging"""
        log_config = self.config.get("logging_and_monitoring", {})
        log_level = log_config.get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_enhanced_systems(self):
        """Initialize enhanced RAG and fusion systems"""
        try:
            # Initialize enhanced RAG retriever
            rag_config = self.config.get("enhanced_rag_config", {})
            self.enhanced_rag = EnhancedRAGRetriever(rag_config)
            
            # Load knowledge bases
            kb_config = self.config.get("knowledge_bases", {})
            cbt_path = kb_config.get("cbt", {}).get("index_path")
            icd11_path = kb_config.get("icd11", {}).get("index_path")
            
            if cbt_path and Path(cbt_path).exists():
                self.enhanced_rag.load_knowledge_bases(cbt_path=cbt_path)
                self.logger.info(f"Loaded CBT knowledge base from {cbt_path}")
                
            if icd11_path and Path(icd11_path).exists():
                self.enhanced_rag.load_knowledge_bases(icd11_path=icd11_path)
                self.logger.info(f"Loaded ICD-11 knowledge base from {icd11_path}")
                
            # Initialize fusion system
            fusion_config = self.config.get("fusion_system_config", {})
            self.fusion_system = IntelligentFusionSystem(fusion_config)
            
            # Connect fusion system with RAG retriever
            if hasattr(self.fusion_system, 'enhanced_retriever'):
                self.fusion_system.enhanced_retriever = self.enhanced_rag
                
            self.system_ready = True
            self.logger.info("Enhanced systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced systems: {e}")
            self.system_ready = False
            
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_model=Dict)
        async def root():
            """Root endpoint with system information"""
            return {
                "service": "Enhanced CBT-ICD11 Mental Health API",
                "version": "2.0.0",
                "status": "operational" if self.system_ready else "limited",
                "features": [
                    "Enhanced RAG Retrieval",
                    "Intelligent Fusion System",
                    "Intent Classification",
                    "Safety Monitoring",
                    "Context Awareness"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        @self.app.get("/health", response_model=SystemStatusResponse)
        async def health_check():
            """Enhanced health check endpoint"""
            kb_status = {"cbt": False, "icd11": False}
            
            if self.enhanced_rag:
                kb_status["cbt"] = self.enhanced_rag.cbt_index is not None
                kb_status["icd11"] = self.enhanced_rag.icd11_index is not None
                
            return SystemStatusResponse(
                enhanced_rag_available=self.enhanced_rag is not None,
                fusion_system_available=self.fusion_system is not None,
                knowledge_bases_loaded=kb_status,
                system_health="healthy" if self.system_ready else "degraded",
                last_updated=datetime.now().isoformat()
            )
            
        @self.app.post("/enhanced_query", response_model=EnhancedQueryResponse)
        async def enhanced_query(request: EnhancedQueryRequest):
            """Enhanced query endpoint with intelligent fusion"""
            if not self.system_ready:
                raise HTTPException(
                    status_code=503, 
                    detail="Enhanced systems not available"
                )
                
            try:
                start_time = time.time()
                
                # Create fusion context
                fusion_context = FusionContext(
                    urgency_level=request.urgency_level,
                    session_stage=request.session_stage,
                    user_history=request.context.get("history", []) if request.context else []
                )
                
                # Process query through fusion system
                fused_response = self.fusion_system.fuse_response(
                    request.message,
                    fusion_context
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Log the interaction
                self.logger.info(
                    f"Enhanced query processed: {request.message[:50]}... "
                    f"-> {fused_response.fusion_strategy.value} "
                    f"({processing_time:.1f}ms)"
                )
                
                return EnhancedQueryResponse(
                    response=fused_response.primary_content,
                    confidence=fused_response.confidence,
                    fusion_strategy=fused_response.fusion_strategy.value,
                    source_breakdown=fused_response.source_breakdown,
                    follow_up_suggestions=fused_response.follow_up_suggestions,
                    safety_notes=fused_response.safety_notes,
                    reasoning=fused_response.reasoning,
                    metadata={
                        **fused_response.metadata,
                        "processing_time_ms": processing_time
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Enhanced query failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/analyze_retrieval", response_model=RetrievalAnalysisResponse)
        async def analyze_retrieval(request: RetrievalAnalysisRequest):
            """Analyze retrieval process for debugging and insights"""
            if not self.enhanced_rag:
                raise HTTPException(
                    status_code=503,
                    detail="Enhanced RAG system not available"
                )
                
            try:
                start_time = time.time()
                
                # Classify intent
                intent, intent_confidence = self.enhanced_rag.intent_classifier.classify_intent(request.query)
                
                # Detect urgency
                urgency_level = self.enhanced_rag.terminology.detect_urgency_level(request.query)
                
                # Perform retrieval
                context = RetrievalContext(urgency_level=urgency_level)
                results = self.enhanced_rag.retrieve(
                    request.query, 
                    context, 
                    top_k=request.max_results
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Format results
                retrieved_sources = []
                for result in results:
                    source_info = {
                        "source": result.source,
                        "relevance_score": result.relevance_score,
                        "confidence": result.confidence,
                        "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
                    }
                    
                    if request.include_reasoning:
                        source_info["reasoning"] = result.reasoning
                        
                    retrieved_sources.append(source_info)
                    
                return RetrievalAnalysisResponse(
                    query=request.query,
                    intent_detected=intent.value,
                    intent_confidence=intent_confidence,
                    urgency_level=urgency_level,
                    retrieved_sources=retrieved_sources,
                    total_results=len(results),
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                self.logger.error(f"Retrieval analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/system_config")
        async def get_system_config():
            """Get current system configuration"""
            if not self.system_ready:
                raise HTTPException(
                    status_code=503,
                    detail="Enhanced systems not available"
                )
                
            # Return sanitized config (remove sensitive information)
            sanitized_config = {
                "enhanced_rag_config": {
                    "retrieval_weights": self.config.get("enhanced_rag_config", {}).get("retrieval_weights", {}),
                    "filtering": self.config.get("enhanced_rag_config", {}).get("filtering", {}),
                    "features": self.config.get("enhanced_rag_config", {}).get("features", {})
                },
                "fusion_system_config": {
                    "decision_tree": self.config.get("fusion_system_config", {}).get("decision_tree", {}),
                    "response_generation": self.config.get("fusion_system_config", {}).get("response_generation", {})
                },
                "intent_classification": self.config.get("intent_classification", {}),
                "safety_monitoring": {
                    "urgency_levels": self.config.get("safety_monitoring", {}).get("urgency_levels", {}),
                    "crisis_resources": self.config.get("safety_monitoring", {}).get("crisis_resources", {})
                }
            }
            
            return sanitized_config
            
        @self.app.post("/safety_check")
        async def safety_check(request: dict):
            """Dedicated safety monitoring endpoint"""
            query = request.get("message", "")
            
            if not query:
                raise HTTPException(status_code=400, detail="Message required")
                
            try:
                # Use fusion system's safety detection
                safety_concern = False
                urgency_level = 1
                
                if self.fusion_system:
                    safety_concern = self.fusion_system._detect_safety_concern(query)
                    
                if self.enhanced_rag:
                    urgency_level = self.enhanced_rag.terminology.detect_urgency_level(query)
                    
                crisis_resources = self.config.get("safety_monitoring", {}).get("crisis_resources", {})
                
                return {
                    "safety_concern_detected": safety_concern,
                    "urgency_level": urgency_level,
                    "crisis_resources": crisis_resources if safety_concern else {},
                    "recommendations": [
                        "Seek immediate professional help" if urgency_level >= 4 else "Consider professional support",
                        "Contact crisis helpline" if safety_concern else "Continue monitoring"
                    ]
                }
                
            except Exception as e:
                self.logger.error(f"Safety check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

# Factory function for easy deployment
def create_enhanced_app(config_path: str = "intelligent_fusion_config.json") -> FastAPI:
    """Create and return configured FastAPI application"""
    integration = EnhancedFastAPIIntegration(config_path)
    return integration.app

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    # Create application
    app = create_enhanced_app()
    
    # Run development server
    print("Starting Enhanced CBT-ICD11 Mental Health API...")
    print("Enhanced features:")
    print("- Intelligent RAG Retrieval")
    print("- Multi-modal Fusion System") 
    print("- Intent Classification")
    print("- Safety Monitoring")
    print("- Context Awareness")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 