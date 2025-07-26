#!/usr/bin/env python3
"""
CBT + ICD-11 System Functionality Demo Test
Complete Demo Test for CBT and ICD-11 Integration System

This script demonstrates the following features:
1. CBT system status check and technique search
2. ICD-11 vector search and medical knowledge retrieval
3. Integrated dialogue flow demonstration
4. End-to-end user interaction simulation
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Add system paths
sys.path.append('CBT_System')
sys.path.append('.')

# Color output support
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_section(title):
    """Print section title"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")

def print_subsection(title):
    """Print subsection title"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'-'*40}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'-'*40}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def test_cbt_system():
    """Test CBT system functionality"""
    print_section("CBT Cognitive Behavioral Therapy System Test")
    
    try:
        from CBT_System.integration import CBTIntegration
        cbt = CBTIntegration(base_dir="CBT_System/cbt_data")
        
        # 1. System status check
        print_subsection("1. CBT System Status Check")
        status = cbt.get_cbt_status()
        
        if status['available']:
            print_success(f"CBT system available")
            print_info(f"CBT technique count: {status['total_techniques']}")
            print_info(f"Content item count: {status['total_content']}")
            print_info(f"Supported categories: {', '.join(status['categories'])}")
        else:
            print_error("CBT system not available")
            return False
            
        # 2. CBT relevance detection test
        print_subsection("2. CBT Relevance Detection Test")
        test_queries = [
            "I feel very anxious and don't know what to do",
            "How to deal with negative thoughts?",
            "I need some relaxation techniques",
            "Thank you for your help",  # Social conversation, should not be relevant
            "What CBT techniques are available?"
        ]
        
        for query in test_queries:
            is_relevant = cbt.should_include_cbt(query)
            status_symbol = "✓" if is_relevant else "✗"
            color = Colors.GREEN if is_relevant else Colors.RED
            print(f"{color}{status_symbol} \"{query}\" - CBT relevant: {is_relevant}{Colors.END}")
        
        # 3. CBT technique search test
        print_subsection("3. CBT Technique Search Test")
        search_queries = [
            "anxiety coping strategies",
            "cognitive restructuring techniques", 
            "relaxation exercises",
            "mindfulness meditation"
        ]
        
        for query in search_queries:
            print_info(f"Search: \"{query}\"")
            results = cbt.cbt_kb.search_cbt_techniques(query, top_k=3)
            
            if results:
                print_success(f"Found {len(results)} relevant techniques:")
                for i, result in enumerate(results[:2], 1):
                    title = result.get('title', 'No title')[:50]
                    score = result.get('relevance_score', 0)
                    category = result.get('category', 'Uncategorized')
                    print(f"  {i}. {title}... (relevance: {score:.3f}, category: {category})")
            else:
                print_warning(f"No relevant techniques found")
            print()
        
        # 4. CBT recommendation system test
        print_subsection("4. CBT Recommendation System Test")
        recommendation_queries = [
            "I always have negative thoughts and can't control them",
            "I feel very stressed and need to relax",
            "How to handle anxiety emotions?"
        ]
        
        for query in recommendation_queries:
            print_info(f"User query: \"{query}\"")
            recommendations = cbt.cbt_kb.get_cbt_recommendation(query)
            
            print(f"  Detected conditions: {', '.join(recommendations['query_analysis']['detected_conditions'])}")
            print(f"  Suggested techniques: {', '.join(recommendations['query_analysis']['suggested_techniques'])}")
            print(f"  Recommended technique count: {len(recommendations['recommended_techniques'])}")
            print(f"  Supporting content count: {len(recommendations['supporting_content'])}")
            
            # Generate formatted response
            if recommendations['recommended_techniques']:
                formatted_response = cbt.cbt_kb.format_cbt_response(recommendations, query)
                print(f"  {Colors.CYAN}Formatted response:{Colors.END}")
                print(f"  \"{formatted_response[:100]}...\"")
            print()
            
        return True
        
    except ImportError as e:
        print_error(f"CBT module import failed: {e}")
        return False
    except Exception as e:
        print_error(f"CBT system test failed: {e}")
        return False

def test_icd11_system():
    """Test ICD-11 system functionality"""
    print_section("ICD-11 Medical Knowledge System Test")
    
    try:
        # Check if FAISS index exists
        if not os.path.exists("embeddings/index.faiss"):
            print_error("ICD-11 FAISS index file does not exist")
            print_warning("Please run generate_faiss_index.py first")
            return False
        
        # Import required modules
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_info(f"Using device: {device}")
        
        # 1. Load embedding model and vector store
        print_subsection("1. Load ICD-11 Vector Database")
        
        try:
            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
            print_success("Embedding model loaded successfully")
            
            store = FAISS.load_local("embeddings", embedder, allow_dangerous_deserialization=True)
            print_success(f"FAISS vector database loaded successfully")
            print_info(f"Vector database size: {store.index.ntotal} documents")
            
        except Exception as e:
            print_error(f"Vector database loading failed: {e}")
            return False
        
        # 2. ICD-11 semantic search test
        print_subsection("2. ICD-11 Semantic Search Test")
        
        search_queries = [
            "depression diagnostic criteria",
            "anxiety disorder classification",
            "bipolar affective disorder",
            "post-traumatic stress disorder",
            "obsessive-compulsive disorder symptoms"
        ]
        
        for query in search_queries:
            print_info(f"Search: \"{query}\"")
            
            try:
                # Execute similarity search
                retriever = store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
                
                if docs:
                    print_success(f"Found {len(docs)} relevant documents:")
                    for i, doc in enumerate(docs, 1):
                        content_preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"  {i}. {content_preview}...")
                        
                        # Show metadata if available
                        if hasattr(doc, 'metadata') and doc.metadata:
                            source = doc.metadata.get('source', 'Unknown source')
                            print(f"     Source: {source}")
                else:
                    print_warning("No relevant documents found")
                    
            except Exception as e:
                print_error(f"Search failed: {e}")
            print()
        
        # 3. Check ICD-11 data files
        print_subsection("3. ICD-11 Data File Check")
        
        if os.path.exists("icd11_ch6_data"):
            csv_file = "icd11_ch6_data/icd11_ch6_entities.csv"
            raw_dir = "icd11_ch6_data/raw"
            
            if os.path.exists(csv_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    print_success(f"CSV entity file: {len(df)} entities")
                    print_info(f"Column names: {', '.join(df.columns.tolist())}")
                except Exception as e:
                    print_warning(f"Cannot read CSV file: {e}")
            
            if os.path.exists(raw_dir):
                json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
                print_success(f"JSON raw data: {len(json_files)} files")
                
                # Check a few sample files
                for i, filename in enumerate(json_files[:3]):
                    filepath = os.path.join(raw_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            print(f"  Example {i+1}: {filename} - {len(str(data))} characters")
                    except Exception as e:
                        print_warning(f"  Cannot read {filename}: {e}")
        else:
            print_warning("ICD-11 data directory does not exist")
        
        return True
        
    except ImportError as e:
        print_error(f"ICD-11 dependency module import failed: {e}")
        return False
    except Exception as e:
        print_error(f"ICD-11 system test failed: {e}")
        return False

def test_integrated_dialogue():
    """Test integrated dialogue flow"""
    print_section("CBT + ICD-11 Integrated Dialogue Flow Test")
    
    # Simulate user dialogue scenarios
    dialogue_scenarios = [
        {
            "name": "Anxiety consultation scenario",
            "user_input": "I've been feeling very anxious lately, can't sleep at night, heart racing, very worried",
            "expected_features": ["ICD-11 anxiety related retrieval", "CBT anxiety coping techniques", "empathetic response"]
        },
        {
            "name": "Depression mood scenario", 
            "user_input": "I feel very depressed, not interested in anything, feel life has no meaning",
            "expected_features": ["ICD-11 depression related retrieval", "CBT behavioral activation techniques", "emotional validation"]
        },
        {
            "name": "Cognitive distortion scenario",
            "user_input": "I always feel I can't do anything right, I'm a failure",
            "expected_features": ["CBT cognitive restructuring techniques", "thought pattern analysis", "balanced thinking guidance"]
        }
    ]
    
    try:
        from CBT_System.integration import CBTIntegration
        cbt = CBTIntegration(base_dir="CBT_System/cbt_data")
        
        for i, scenario in enumerate(dialogue_scenarios, 1):
            print_subsection(f"{i}. {scenario['name']}")
            
            user_input = scenario['user_input']
            print_info(f"User input: \"{user_input}\"")
            
            # Check CBT relevance
            is_cbt_relevant = cbt.should_include_cbt(user_input)
            print(f"CBT relevance: {Colors.GREEN if is_cbt_relevant else Colors.RED}{is_cbt_relevant}{Colors.END}")
            
            if is_cbt_relevant:
                # Get CBT recommendations
                recommendations = cbt.cbt_kb.get_cbt_recommendation(user_input)
                print_success("CBT analysis results:")
                print(f"  Detected conditions: {', '.join(recommendations['query_analysis']['detected_conditions'])}")
                print(f"  Recommended techniques: {', '.join(recommendations['query_analysis']['suggested_techniques'])}")
                
                # Generate CBT enhanced response
                context = f"User expressed the following situation: {user_input}"
                base_response = "I understand your feelings, this is indeed a situation that needs attention."
                
                enhanced_response = cbt.enhance_response_with_cbt(
                    user_query=user_input,
                    context=context,
                    base_response=base_response
                )
                
                print_success("CBT enhanced response:")
                print(f"  {Colors.CYAN}\"{enhanced_response[:150]}...\"{Colors.END}")
            else:
                print_warning("This input is not suitable for CBT enhancement")
            
            print()
        
        return True
        
    except Exception as e:
        print_error(f"Integrated dialogue test failed: {e}")
        return False

def test_fastapi_endpoints():
    """Test FastAPI endpoint functionality"""
    print_section("FastAPI Endpoint Functionality Test")
    
    try:
        import requests
        import time
        
        base_url = "http://localhost:8000"
        
        # 1. Health check
        print_subsection("1. System Health Check")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print_success("FastAPI server running normally")
                print_info(f"Device: {health_data.get('device', 'N/A')}")
                print_info(f"LLM loaded: {health_data.get('psychologist_llm_loaded', False)}")
                print_info(f"Vector store loaded: {health_data.get('store_loaded', False)}")
                print_info(f"CBT available: {health_data.get('cbt_available', False)}")
                print_info(f"Interaction count: {health_data.get('interactions_count', 0)}")
            else:
                print_error(f"Health check failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print_error(f"Cannot connect to FastAPI server: {e}")
            print_warning("Please ensure running: python fastapi_server.py")
            return False
        
        # 2. Dialogue API test
        print_subsection("2. Dialogue API Test")
        
        test_messages = [
            "I feel very anxious, is there any way to help me?",
            "How to handle negative thoughts?",
            "I haven't been sleeping well lately, affecting my mood"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print_info(f"Test message {i}: \"{message}\"")
            
            payload = {
                "question": message,
                "type": "empathetic_professional",
                "history": []
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/api/empathetic_professional",
                    json=payload,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get('answer', '')
                    context_used = data.get('context_used', '')
                    prompt_source = data.get('prompt_source', 'unknown')
                    
                    print_success(f"Response generated successfully (time: {response_time:.2f}s)")
                    print(f"  Prompt source: {prompt_source}")
                    print(f"  Response length: {len(answer)} characters")
                    print(f"  {Colors.CYAN}Response content: \"{answer[:100]}...\"{Colors.END}")
                    
                    if context_used:
                        print(f"  Context used: {len(context_used)} characters")
                else:
                    print_error(f"API call failed: HTTP {response.status_code}")
                    print(f"Error message: {response.text}")
                    
            except requests.exceptions.Timeout:
                print_error("Request timeout")
            except requests.exceptions.RequestException as e:
                print_error(f"Request failed: {e}")
            
            print()
        
        # 3. CBT endpoint test
        print_subsection("3. CBT Dedicated Endpoint Test")
        
        try:
            # CBT status check
            response = requests.get(f"{base_url}/api/cbt/status", timeout=5)
            if response.status_code == 200:
                cbt_status = response.json()
                print_success("CBT status check successful")
                print_info(f"CBT available: {cbt_status.get('cbt_available', False)}")
                
                if cbt_status.get('cbt_available'):
                    status_data = cbt_status.get('status', {})
                    print_info(f"Technique count: {status_data.get('total_techniques', 0)}")
                    print_info(f"Content count: {status_data.get('total_content', 0)}")
            else:
                print_warning("CBT status check failed")
            
            # CBT recommendation test
            cbt_payload = {
                "query": "I need some techniques to cope with anxiety",
                "context": "User expressed anxiety emotions"
            }
            
            response = requests.post(
                f"{base_url}/api/cbt/recommend",
                json=cbt_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                cbt_data = response.json()
                print_success("CBT recommendation successful")
                print_info(f"Relevance: {cbt_data.get('cbt_relevant', False)}")
                print_info(f"Recommended technique count: {len(cbt_data.get('recommended_techniques', []))}")
                print(f"  {Colors.CYAN}Recommendation response: \"{cbt_data.get('formatted_response', '')[:100]}...\"{Colors.END}")
            else:
                print_warning(f"CBT recommendation failed: HTTP {response.status_code}")
                
        except Exception as e:
            print_warning(f"CBT endpoint test failed: {e}")
        
        return True
        
    except ImportError as e:
        print_error(f"requests module not installed: {e}")
        print_info("Please run: pip install requests")
        return False

def generate_test_report():
    """Generate test report"""
    print_section("Test Report Generation")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    # Run all tests
    test_results = {
        "timestamp": timestamp,
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "test_results": {}
    }
    
    print_info("Running complete test suite...")
    
    # CBT system test
    print_info("Testing CBT system...")
    cbt_result = test_cbt_system()
    test_results["test_results"]["cbt_system"] = {
        "passed": cbt_result,
        "timestamp": datetime.now().isoformat()
    }
    
    # ICD-11 system test  
    print_info("Testing ICD-11 system...")
    icd11_result = test_icd11_system()
    test_results["test_results"]["icd11_system"] = {
        "passed": icd11_result,
        "timestamp": datetime.now().isoformat()
    }
    
    # Integrated dialogue test
    print_info("Testing integrated dialogue...")
    dialogue_result = test_integrated_dialogue()
    test_results["test_results"]["integrated_dialogue"] = {
        "passed": dialogue_result,
        "timestamp": datetime.now().isoformat()
    }
    
    # API endpoint test
    print_info("Testing API endpoints...")
    api_result = test_fastapi_endpoints()
    test_results["test_results"]["fastapi_endpoints"] = {
        "passed": api_result,
        "timestamp": datetime.now().isoformat()
    }
    
    # Calculate overall results
    total_tests = len(test_results["test_results"])
    passed_tests = sum(1 for result in test_results["test_results"].values() if result["passed"])
    
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0
    }
    
    # Save report
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print_success(f"Test report saved: {report_file}")
    except Exception as e:
        print_error(f"Failed to save test report: {e}")
    
    # Display summary
    print_subsection("Test Summary")
    success_rate = test_results["summary"]["success_rate"]
    color = Colors.GREEN if success_rate >= 0.8 else Colors.YELLOW if success_rate >= 0.5 else Colors.RED
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {total_tests - passed_tests}")
    print(f"{color}Success rate: {success_rate:.1%}{Colors.END}")
    
    return test_results

def main():
    """Main function"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("🧠 CBT + ICD-11 Mental Health Consultation System")
    print("   Functionality Demo Test Program")
    print("   ===================================")
    print(f"{Colors.END}")
    
    print_info("This program will test the following functional modules:")
    print("  1. CBT Cognitive Behavioral Therapy System")
    print("  2. ICD-11 Medical Knowledge System") 
    print("  3. Integrated Dialogue Flow")
    print("  4. FastAPI Endpoint Functionality")
    print("  5. Generate Detailed Test Report")
    
    print(f"\n{Colors.YELLOW}Please ensure the following conditions:{Colors.END}")
    print("  ✓ CBT data has been collected and processed")
    print("  ✓ ICD-11 vector index has been generated")
    print("  ✓ FastAPI server is running (optional)")
    print("  ✓ Required Python dependencies are installed")
    
    input(f"\n{Colors.BOLD}Press Enter to start testing...{Colors.END}")
    
    try:
        # Select test mode
        print(f"\n{Colors.BOLD}Select test mode:{Colors.END}")
        print("1. Complete test (recommended)")
        print("2. Test CBT system only")
        print("3. Test ICD-11 system only")
        print("4. Test integrated dialogue only")
        print("5. Test API endpoints only")
        print("6. Generate complete test report")
        
        choice = input("Please enter choice (1-6): ").strip()
        
        if choice == "1":
            # Complete test
            test_cbt_system()
            test_icd11_system() 
            test_integrated_dialogue()
            test_fastapi_endpoints()
            
        elif choice == "2":
            test_cbt_system()
            
        elif choice == "3":
            test_icd11_system()
            
        elif choice == "4":
            test_integrated_dialogue()
            
        elif choice == "5":
            test_fastapi_endpoints()
            
        elif choice == "6":
            generate_test_report()
            
        else:
            print_error("Invalid choice, running default complete test")
            test_cbt_system()
            test_icd11_system()
            test_integrated_dialogue()
        
        print_section("Test Complete")
        print_success("All selected tests have been completed!")
        print_info("Check the results above to understand the status of each module")
        
        # Provide suggestions
        print(f"\n{Colors.BOLD}Suggested next steps:{Colors.END}")
        print("  • If any tests failed, please check dependencies and data files")
        print("  • Run FastAPI server for complete experience")
        print("  • Use Web interface for user interaction testing")
        print("  • Check generated test reports for detailed information")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Testing interrupted by user{Colors.END}")
    except Exception as e:
        print_error(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 