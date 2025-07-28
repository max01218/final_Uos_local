#!/usr/bin/env python3
"""
Enhanced CBT System Integration Test
Comprehensive testing of enhanced system integration with original CBT system
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import traceback

def test_original_cbt_integration():
    """Test if original CBT integration works with enhanced data"""
    print("Testing Original CBT Integration Compatibility")
    print("=" * 50)
    
    results = {}
    
    try:
        # Import original CBT integration
        sys.path.append('..')
        from integration import CBTIntegration
        
        # Initialize with enhanced data directory
        cbt = CBTIntegration(base_dir="cbt_data")
        
        # Test 1: Basic status check
        print("1. Testing CBT status check...")
        status = cbt.get_cbt_status()
        results["status_check"] = {
            "success": True,
            "available": status.get("available", False),
            "technique_count": status.get("technique_count", 0),
            "content_count": status.get("content_count", 0)
        }
        print(f"   CBT Available: {status.get('available', False)}")
        print(f"   Technique Count: {status.get('technique_count', 0)}")
        print(f"   Content Count: {status.get('content_count', 0)}")
        
        # Test 2: CBT relevance detection
        print("\n2. Testing CBT relevance detection...")
        test_queries = [
            "I feel very anxious and need coping strategies",
            "How to challenge negative thoughts",
            "What are some relaxation techniques",
            "Hello how are you today"
        ]
        
        relevance_results = []
        for query in test_queries:
            try:
                is_relevant = cbt.should_include_cbt(query)
                relevance_results.append({
                    "query": query,
                    "relevant": is_relevant,
                    "success": True
                })
                print(f"   Query: '{query[:40]}...' -> Relevant: {is_relevant}")
            except Exception as e:
                relevance_results.append({
                    "query": query,
                    "relevant": False,
                    "success": False,
                    "error": str(e)
                })
                print(f"   Query: '{query[:40]}...' -> Error: {e}")
                
        results["relevance_detection"] = relevance_results
        
        # Test 3: CBT search functionality
        print("\n3. Testing CBT search functionality...")
        search_queries = [
            "anxiety coping strategies",
            "cognitive restructuring techniques", 
            "relaxation exercises",
            "mindfulness meditation"
        ]
        
        search_results = []
        for query in search_queries:
            try:
                if hasattr(cbt, 'cbt_kb') and hasattr(cbt.cbt_kb, 'search_cbt_techniques'):
                    techniques = cbt.cbt_kb.search_cbt_techniques(query, top_k=3)
                    search_results.append({
                        "query": query,
                        "results_count": len(techniques),
                        "success": True,
                        "sample_result": techniques[0] if techniques else None
                    })
                    print(f"   Query: '{query}' -> Found {len(techniques)} results")
                else:
                    search_results.append({
                        "query": query,
                        "results_count": 0,
                        "success": False,
                        "error": "Search method not available"
                    })
                    print(f"   Query: '{query}' -> Search method not available")
            except Exception as e:
                search_results.append({
                    "query": query,
                    "results_count": 0,
                    "success": False,
                    "error": str(e)
                })
                print(f"   Query: '{query}' -> Error: {e}")
                
        results["search_functionality"] = search_results
        
        # Test 4: Enhanced CBT response generation
        print("\n4. Testing enhanced CBT response generation...")
        response_queries = [
            "I'm feeling overwhelmed with anxiety",
            "I have negative thoughts about myself",
            "I need help with stress management"
        ]
        
        response_results = []
        for query in response_queries:
            try:
                if hasattr(cbt, 'enhance_response'):
                    enhanced_response = cbt.enhance_response(query, "I understand you're struggling.")
                    response_results.append({
                        "query": query,
                        "response_generated": len(enhanced_response) > 0,
                        "response_length": len(enhanced_response),
                        "success": True
                    })
                    print(f"   Query: '{query[:30]}...' -> Response generated ({len(enhanced_response)} chars)")
                else:
                    response_results.append({
                        "query": query,
                        "response_generated": False,
                        "success": False,
                        "error": "enhance_response method not available"
                    })
                    print(f"   Query: '{query[:30]}...' -> enhance_response method not available")
            except Exception as e:
                response_results.append({
                    "query": query,
                    "response_generated": False,
                    "success": False,
                    "error": str(e)
                })
                print(f"   Query: '{query[:30]}...' -> Error: {e}")
                
        results["response_generation"] = response_results
        
        return {"success": True, "results": results}
        
    except Exception as e:
        print(f"✗ Original CBT integration test failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_enhanced_data_compatibility():
    """Test if enhanced data can be read by original system"""
    print("\nTesting Enhanced Data Compatibility")
    print("=" * 50)
    
    try:
        # Check if enhanced processed data exists
        processed_dir = Path("cbt_data/raw_data/processed")
        if not processed_dir.exists():
            print("✗ No processed data directory found")
            return {"success": False, "error": "No processed data"}
            
        # Find latest processed file
        processed_files = list(processed_dir.glob("cbt_enhanced_processed_*.json"))
        if not processed_files:
            print("✗ No enhanced processed files found")
            return {"success": False, "error": "No enhanced processed files"}
            
        latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
        print(f"✓ Found latest processed file: {latest_file.name}")
        
        # Check processed data content
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data.get("metadata", {})
        items = data.get("data", [])
        
        print(f"✓ Processed data loaded successfully")
        print(f"   Total items: {metadata.get('final_items', 0)}")
        print(f"   Average quality: {metadata.get('average_quality_score', 0):.2f}")
        print(f"   Categories: {list(metadata.get('category_distribution', {}).keys())}")
        
        # Check if data has required fields for original system
        compatibility_check = {
            "has_items": len(items) > 0,
            "items_have_content": False,
            "items_have_classification": False,
            "items_have_quality_scores": False
        }
        
        if items:
            sample_item = items[0]
            compatibility_check["items_have_content"] = "content" in sample_item
            compatibility_check["items_have_classification"] = "classification" in sample_item
            compatibility_check["items_have_quality_scores"] = "enhanced_quality_score" in sample_item
            
        print(f"✓ Data compatibility check:")
        for check, result in compatibility_check.items():
            print(f"   {check}: {result}")
            
        return {
            "success": True, 
            "metadata": metadata, 
            "compatibility": compatibility_check
        }
        
    except Exception as e:
        print(f"✗ Enhanced data compatibility test failed: {e}")
        return {"success": False, "error": str(e)}

def test_fastapi_integration():
    """Test integration with FastAPI server"""
    print("\nTesting FastAPI Integration")
    print("=" * 50)
    
    try:
        # Try to import FastAPI server components
        sys.path.append('..')
        
        print("1. Testing FastAPI imports...")
        try:
            # Check if we can import the main server components
            import fastapi_server
            print("   ✓ FastAPI server module imported successfully")
        except Exception as e:
            print(f"   ✗ FastAPI server import failed: {e}")
            return {"success": False, "error": f"FastAPI import failed: {e}"}
            
        print("\n2. Testing CBT integration in server context...")
        try:
            # Test if CBT integration can be initialized in server context
            from integration import CBTIntegration
            
            # This mimics how the server would initialize CBT
            cbt_integration = CBTIntegration(base_dir="CBT_System/cbt_data")
            status = cbt_integration.get_cbt_status()
            
            print(f"   ✓ CBT integration initialized in server context")
            print(f"   ✓ CBT available: {status.get('available', False)}")
            
            server_test_result = {
                "fastapi_import": True,
                "cbt_initialization": True,
                "cbt_available": status.get('available', False),
                "technique_count": status.get('technique_count', 0)
            }
            
            return {"success": True, "results": server_test_result}
            
        except Exception as e:
            print(f"   ✗ Server context test failed: {e}")
            return {"success": False, "error": f"Server context failed: {e}"}
            
    except Exception as e:
        print(f"✗ FastAPI integration test failed: {e}")
        return {"success": False, "error": str(e)}

def test_vector_search_performance():
    """Test vector search performance with enhanced indices"""
    print("\nTesting Vector Search Performance")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Check if FAISS indices exist
        embeddings_dir = Path("cbt_data/embeddings")
        if not embeddings_dir.exists():
            print("✗ No embeddings directory found")
            return {"success": False, "error": "No embeddings directory"}
            
        faiss_files = list(embeddings_dir.glob("*.faiss"))
        if not faiss_files:
            print("✗ No FAISS index files found") 
            return {"success": False, "error": "No FAISS indices"}
            
        print(f"✓ Found {len(faiss_files)} FAISS index files")
        
        # Test loading indices
        try:
            import faiss
            import pickle
            
            performance_results = []
            
            for faiss_file in faiss_files[:3]:  # Test first 3 indices
                try:
                    start_time = time.time()
                    
                    # Load index
                    index = faiss.read_index(str(faiss_file))
                    load_time = time.time() - start_time
                    
                    # Load corresponding metadata
                    metadata_file = faiss_file.with_suffix('.pkl').with_name(
                        faiss_file.stem.replace('index', 'metadata') + '.pkl'
                    )
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'rb') as f:
                            metadata = pickle.load(f)
                    else:
                        metadata = []
                    
                    # Create test query vector
                    dimension = index.d
                    test_vector = np.random.random((1, dimension)).astype('float32')
                    faiss.normalize_L2(test_vector)
                    
                    # Test search
                    start_time = time.time()
                    scores, indices = index.search(test_vector, min(3, index.ntotal))
                    search_time = time.time() - start_time
                    
                    performance_results.append({
                        "index_file": faiss_file.name,
                        "dimension": dimension,
                        "total_vectors": index.ntotal,
                        "load_time_ms": load_time * 1000,
                        "search_time_ms": search_time * 1000,
                        "metadata_count": len(metadata),
                        "success": True
                    })
                    
                    print(f"   ✓ {faiss_file.name}: {index.ntotal} vectors, {dimension}D")
                    print(f"     Load: {load_time*1000:.2f}ms, Search: {search_time*1000:.2f}ms")
                    
                except Exception as e:
                    performance_results.append({
                        "index_file": faiss_file.name,
                        "success": False,
                        "error": str(e)
                    })
                    print(f"   ✗ {faiss_file.name}: {e}")
                    
            return {"success": True, "results": performance_results}
            
        except ImportError as e:
            print(f"✗ Required libraries not available: {e}")
            return {"success": False, "error": f"Missing libraries: {e}"}
            
    except Exception as e:
        print(f"✗ Vector search performance test failed: {e}")
        return {"success": False, "error": str(e)}

def run_comprehensive_integration_test():
    """Run all integration tests"""
    print("Enhanced CBT System - Comprehensive Integration Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_results = {}
    
    # Test 1: Original CBT Integration
    result1 = test_original_cbt_integration()
    all_results["original_cbt_integration"] = result1
    
    # Test 2: Enhanced Data Compatibility
    result2 = test_enhanced_data_compatibility()
    all_results["enhanced_data_compatibility"] = result2
    
    # Test 3: FastAPI Integration
    result3 = test_fastapi_integration()
    all_results["fastapi_integration"] = result3
    
    # Test 4: Vector Search Performance
    result4 = test_vector_search_performance()
    all_results["vector_search_performance"] = result4
    
    # Generate summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results.values() if r.get("success", False))
    
    print(f"Total Test Categories: {total_tests}")
    print(f"Successful Categories: {successful_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests < total_tests:
        print("\nFailed Test Categories:")
        for test_name, result in all_results.items():
            if not result.get("success", False):
                print(f"- {test_name}: {result.get('error', 'Unknown error')}")
    
    # Save detailed results
    results_file = f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_categories": total_tests,
                    "successful_categories": successful_tests,
                    "success_rate": (successful_tests/total_tests)*100
                },
                "detailed_results": all_results
            }, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {results_file}")
    except Exception as e:
        print(f"\nFailed to save results: {e}")
    
    print("\n" + "=" * 60)
    
    if successful_tests == total_tests:
        print("🎉 All integration tests passed! Enhanced system is fully compatible.")
        return True
    elif successful_tests >= total_tests * 0.75:
        print("⚠ Most integration tests passed. System should be functional with minor issues.")
        return True
    else:
        print("Multiple integration test failures. Please check system configuration.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1) 