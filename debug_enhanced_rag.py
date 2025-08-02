#!/usr/bin/env python3
"""
Debug Enhanced RAG System
"""

import sys
import os
sys.path.append('.')

try:
    from enhanced_rag_retriever import EnhancedRAGRetriever
    from intelligent_fusion_system import IntelligentFusionSystem
    import json
    import faiss
    import pickle
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def debug_enhanced_rag():
    print("Debugging Enhanced RAG System")
    print("="*50)
    
    # 1. Check config file
    config_path = "intelligent_fusion_config.json"
    print(f"\n1. Checking config file: {config_path}")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Config file loaded successfully")
        print(f"   Enhanced RAG config keys: {list(config.get('enhanced_rag_config', {}).keys())}")
    else:
        print(f"❌ Config file not found")
        return
    
    # 2. Check ICD-11 data files
    print(f"\n2. Checking ICD-11 data files")
    icd11_index_path = "embeddings/index.faiss"
    icd11_metadata_path = "embeddings/index.pkl"
    
    if os.path.exists(icd11_index_path):
        print(f"✅ ICD-11 index exists: {icd11_index_path}")
        try:
            index = faiss.read_index(icd11_index_path)
            print(f"   Index size: {index.ntotal}")
        except Exception as e:
            print(f"   ❌ Error reading index: {e}")
    else:
        print(f"❌ ICD-11 index not found: {icd11_index_path}")
    
    if os.path.exists(icd11_metadata_path):
        print(f"✅ ICD-11 metadata exists: {icd11_metadata_path}")
        try:
            with open(icd11_metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"   Metadata type: {type(metadata)}")
            if isinstance(metadata, list):
                print(f"   Metadata length: {len(metadata)}")
            elif isinstance(metadata, tuple):
                print(f"   Metadata tuple length: {len(metadata)}")
        except Exception as e:
            print(f"   ❌ Error reading metadata: {e}")
    else:
        print(f"❌ ICD-11 metadata not found: {icd11_metadata_path}")
    
    # 3. Check CBT data files
    print(f"\n3. Checking CBT data files")
    cbt_index_path = "CBT_System/cbt_data/embeddings/cbt_index_standard_20250727_143814.faiss"
    cbt_metadata_path = "CBT_System/cbt_data/embeddings/cbt_metadata_standard_20250727_143814.pkl"
    
    if os.path.exists(cbt_index_path):
        print(f"✅ CBT index exists: {cbt_index_path}")
        try:
            index = faiss.read_index(cbt_index_path)
            print(f"   Index size: {index.ntotal}")
        except Exception as e:
            print(f"   ❌ Error reading index: {e}")
    else:
        print(f"❌ CBT index not found: {cbt_index_path}")
    
    if os.path.exists(cbt_metadata_path):
        print(f"✅ CBT metadata exists: {cbt_metadata_path}")
        try:
            with open(cbt_metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"   Metadata type: {type(metadata)}")
            if isinstance(metadata, list):
                print(f"   Metadata length: {len(metadata)}")
        except Exception as e:
            print(f"   ❌ Error reading metadata: {e}")
    else:
        print(f"❌ CBT metadata not found: {cbt_metadata_path}")
    
    # 4. Test Enhanced RAG Retriever initialization
    print(f"\n4. Testing Enhanced RAG Retriever initialization")
    try:
        enhanced_rag = EnhancedRAGRetriever(config['enhanced_rag_config'])
        print(f"✅ Enhanced RAG Retriever initialized successfully")
        
        # Test knowledge base loading
        enhanced_rag.load_knowledge_bases()
        print(f"✅ Knowledge bases loaded")
        
        # Test retrieval
        print(f"\n5. Testing retrieval with 'what is anxiety'")
        from enhanced_rag_retriever import RetrievalContext, RetrievalStrategy, QueryIntent
        
        context = RetrievalContext(
            conversation_history=[],
            emotional_state="neutral",
            urgency_level=1
        )
        
        results = enhanced_rag.retrieve(
            query="what is anxiety",
            context=context,
            strategy=RetrievalStrategy.HYBRID_SEMANTIC_KEYWORD,
            top_k=3
        )
        
        print(f"✅ Retrieval successful")
        print(f"   Number of results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Source: {result.source}")
            print(f"   Relevance score: {result.relevance_score:.3f}")
            print(f"   Content preview: {result.content[:200]}...")
            
    except Exception as e:
        print(f"❌ Enhanced RAG Retriever error: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Test Intelligent Fusion System
    print(f"\n6. Testing Intelligent Fusion System")
    try:
        fusion_system = IntelligentFusionSystem(config['fusion_system_config'])
        print(f"✅ Intelligent Fusion System initialized successfully")
        
        # Test fusion
        from intelligent_fusion_system import FusionContext, QueryIntent
        
        fusion_context = FusionContext(
            urgency_level=1,
            user_history=[],
            session_stage="ongoing"
        )
        
        fused_response = fusion_system.fuse_response(
            query="what is anxiety",
            context=fusion_context
        )
        
        print(f"✅ Fusion successful")
        print(f"   Strategy: {fused_response.fusion_strategy.value}")
        print(f"   Confidence: {fused_response.confidence:.3f}")
        print(f"   Primary content length: {len(fused_response.primary_content)}")
        print(f"   Supporting content length: {len(fused_response.supporting_content)}")
        
    except Exception as e:
        print(f"❌ Intelligent Fusion System error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*50)
    print("Debug Complete")

if __name__ == "__main__":
    debug_enhanced_rag() 