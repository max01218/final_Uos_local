#!/usr/bin/env python3
"""
Complete diagnosis of FastAPI server issues
"""

import sys
import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_imports():
    """Check if all required modules can be imported"""
    print("=== Checking Imports ===")
    
    try:
        from enhanced_rag_retriever import EnhancedRAGRetriever, RetrievalContext, RetrievalStrategy
        print("✓ EnhancedRAGRetriever imported successfully")
    except Exception as e:
        print(f"✗ Failed to import EnhancedRAGRetriever: {e}")
        return False
    
    try:
        from intelligent_fusion_system import IntelligentFusionSystem, FusionContext
        print("✓ IntelligentFusionSystem imported successfully")
    except Exception as e:
        print(f"✗ Failed to import IntelligentFusionSystem: {e}")
        return False
    
    try:
        from CBT_System.integration import CBTIntegration
        print("✓ CBTIntegration imported successfully")
    except Exception as e:
        print(f"✗ Failed to import CBTIntegration: {e}")
        return False
    
    return True

def check_config_files():
    """Check if configuration files exist"""
    print("\n=== Checking Configuration Files ===")
    
    config_files = [
        "intelligent_fusion_config.json",
        "OPRO_Streamlined/prompts/optimized_prompt.txt",
        "ICD11_OPRO/prompts/optimized_prompt.txt"
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} missing")
            all_exist = False
    
    return all_exist

def check_data_files():
    """Check if data files exist"""
    print("\n=== Checking Data Files ===")
    
    data_files = [
        "CBT_System/cbt_data/embeddings/cbt_index.faiss",
        "CBT_System/cbt_data/embeddings/cbt_metadata.pkl",
        "embeddings/index.faiss",
        "embeddings/index.pkl"
    ]
    
    all_exist = True
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"✓ {data_file} exists")
        else:
            print(f"✗ {data_file} missing")
            all_exist = False
    
    return all_exist

def test_enhanced_systems_initialization():
    """Test enhanced systems initialization"""
    print("\n=== Testing Enhanced Systems Initialization ===")
    
    try:
        from enhanced_rag_retriever import EnhancedRAGRetriever
        from intelligent_fusion_system import IntelligentFusionSystem
        
        # Load configuration
        config_path = Path("intelligent_fusion_config.json")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            print("✗ Config file not found")
            return False
        
        # Initialize Enhanced RAG Retriever
        enhanced_rag_config = config.get("enhanced_rag_config", {})
        flattened_config = {
            "semantic_weight": enhanced_rag_config.get("retrieval_weights", {}).get("semantic_weight", 0.5),
            "keyword_weight": enhanced_rag_config.get("retrieval_weights", {}).get("keyword_weight", 0.3),
            "context_weight": enhanced_rag_config.get("retrieval_weights", {}).get("context_weight", 0.2),
            "intent_boost": enhanced_rag_config.get("retrieval_weights", {}).get("intent_boost", 0.15),
            "urgency_boost": enhanced_rag_config.get("retrieval_weights", {}).get("urgency_boost", 0.1),
            "min_relevance_threshold": enhanced_rag_config.get("filtering", {}).get("min_relevance_threshold", 0.3),
            "max_results": enhanced_rag_config.get("filtering", {}).get("max_results", 5),
            "enable_query_expansion": enhanced_rag_config.get("features", {}).get("enable_query_expansion", True),
            "enable_intent_filtering": enhanced_rag_config.get("features", {}).get("enable_intent_filtering", True),
            "enable_context_awareness": enhanced_rag_config.get("features", {}).get("enable_context_awareness", True)
        }
        
        print(f"✓ Flattened config created: {flattened_config}")
        
        enhanced_rag_retriever = EnhancedRAGRetriever(flattened_config)
        print("✓ EnhancedRAGRetriever initialized")
        
        # Test knowledge base loading
        cbt_index_path = "CBT_System/cbt_data/embeddings/cbt_index.faiss"
        icd11_index_path = "embeddings/index.faiss"
        
        enhanced_rag_retriever.load_knowledge_bases(cbt_index_path, icd11_index_path)
        print("✓ Knowledge bases loaded")
        
        # Initialize Intelligent Fusion System
        intelligent_fusion_system = IntelligentFusionSystem(config.get("fusion_system_config", {}))
        intelligent_fusion_system.enhanced_retriever = enhanced_rag_retriever
        intelligent_fusion_system.initialize_knowledge_bases(cbt_index_path, icd11_index_path)
        print("✓ IntelligentFusionSystem initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced systems initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_functionality():
    """Test retrieval functionality"""
    print("\n=== Testing Retrieval Functionality ===")
    
    try:
        from enhanced_rag_retriever import EnhancedRAGRetriever, RetrievalContext, RetrievalStrategy
        
        # Create a simple retriever
        retriever = EnhancedRAGRetriever()
        
        # Test retrieve method
        query = "I feel anxious"
        context = RetrievalContext()
        
        results = retriever.retrieve(
            query=query,
            context=context,
            strategy=RetrievalStrategy.HYBRID_SEMANTIC_KEYWORD,
            top_k=5
        )
        
        print(f"✓ Retrieval successful, got {len(results)} results")
        return True
        
    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_global_variables():
    """Check global variable handling"""
    print("\n=== Checking Global Variable Handling ===")
    
    # Simulate the global variable issue
    enhanced_rag_retriever = None
    intelligent_fusion_system = None
    
    def test_function():
        global enhanced_rag_retriever, intelligent_fusion_system
        
        # This should work
        if enhanced_rag_retriever and intelligent_fusion_system:
            print("This should not execute")
        
        # This should also work
        if globals().get('enhanced_rag_retriever') and globals().get('intelligent_fusion_system'):
            print("This should not execute")
        
        # This should work
        enhanced_rag_retriever = globals().get('enhanced_rag_retriever')
        if enhanced_rag_retriever:
            print("This should not execute")
    
    try:
        test_function()
        print("✓ Global variable handling works correctly")
        return True
    except Exception as e:
        print(f"✗ Global variable handling failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("FastAPI Server Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("Imports", check_imports),
        ("Configuration Files", check_config_files),
        ("Data Files", check_data_files),
        ("Enhanced Systems Initialization", test_enhanced_systems_initialization),
        ("Retrieval Functionality", test_retrieval_functionality),
        ("Global Variable Handling", check_global_variables)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The server should work correctly.")
    else:
        print("✗ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 