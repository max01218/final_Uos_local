#!/usr/bin/env python3
"""
CBT + ICD-11 Quick Feature Demo
Quick Demo for CBT and ICD-11 Integration

Simplified version of the feature demo, focusing on core functionality:
1. CBT technique recommendation and search
2. ICD-11 medical knowledge retrieval
3. Integrated dialogue flow demonstration
"""

import sys
import os
sys.path.append('CBT_System')
sys.path.append('.')

def demo_cbt_features():
    """Demo CBT functionality"""
    print("\n" + "="*50)
    print("🧠 CBT Cognitive Behavioral Therapy System Demo")
    print("="*50)
    
    try:
        from CBT_System.integration import CBTIntegration
        cbt = CBTIntegration()
        
        # Check CBT system status
        status = cbt.get_cbt_status()
        print(f"✅ CBT system status: {'Available' if status['available'] else 'Not available'}")
        
        if not status['available']:
            print("❌ CBT system not available, please run CBT data collection and processing first")
            return
            
        print(f"📊 CBT technique count: {status['total_techniques']}")
        print(f"📊 Content item count: {status['total_content']}")
        print(f"📋 Supported categories: {', '.join(status['categories'])}")
        
        # Demo CBT relevance detection
        print("\n🔍 CBT Relevance Detection Demo:")
        test_inputs = [
            "I feel very anxious and don't know what to do",
            "How to deal with negative thoughts?", 
            "Thank you for your help",  # Social conversation
        ]
        
        for input_text in test_inputs:
            is_relevant = cbt.should_include_cbt(input_text)
            symbol = "✅" if is_relevant else "❌"
            print(f"  {symbol} \"{input_text}\" → CBT relevant: {is_relevant}")
        
        # Demo CBT technique search
        print("\n🔎 CBT Technique Search Demo:")
        search_query = "anxiety coping strategies"
        results = cbt.cbt_kb.search_cbt_techniques(search_query, top_k=2)
        print(f"Search: \"{search_query}\"")
        
        if results:
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')[:60]
                score = result.get('relevance_score', 0)
                print(f"  {i}. {title}... (relevance: {score:.3f})")
        else:
            print("  No relevant techniques found")
        
        # Demo CBT recommendation system
        print("\n💡 CBT Recommendation System Demo:")
        user_query = "I always have negative thoughts and can't control them"
        print(f"User query: \"{user_query}\"")
        
        recommendations = cbt.cbt_kb.get_cbt_recommendation(user_query)
        conditions = recommendations['query_analysis']['detected_conditions']
        techniques = recommendations['query_analysis']['suggested_techniques']
        
        print(f"  Detected conditions: {', '.join(conditions) if conditions else 'No specific conditions'}")
        print(f"  Suggested techniques: {', '.join(techniques) if techniques else 'General support'}")
        
        if recommendations['recommended_techniques']:
            formatted_response = cbt.cbt_kb.format_cbt_response(recommendations, user_query)
            print(f"  💬 CBT response example:")
            print(f"    \"{formatted_response[:120]}...\"")
        
        return True
        
    except ImportError:
        print("❌ Cannot import CBT module, please check if CBT_System is properly installed")
        return False
    except Exception as e:
        print(f"❌ CBT demo failed: {e}")
        return False

def demo_icd11_features():
    """Demo ICD-11 functionality"""
    print("\n" + "="*50)
    print("🏥 ICD-11 Medical Knowledge System Demo")
    print("="*50)
    
    try:
        # Check vector index file
        if not os.path.exists("embeddings/index.faiss"):
            print("❌ ICD-11 vector index file does not exist")
            print("💡 Please run: python generate_faiss_index.py")
            return False
        
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        # Load vector database
        print("📚 Loading ICD-11 knowledge base...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
        )
        
        store = FAISS.load_local("embeddings", embedder, allow_dangerous_deserialization=True)
        print(f"✅ ICD-11 vector database loaded successfully ({store.index.ntotal} documents)")
        
        # Demo semantic search
        print("\n🔍 ICD-11 Semantic Search Demo:")
        search_queries = [
            "depression diagnostic criteria",
            "anxiety disorder classification",
            "post-traumatic stress disorder"
        ]
        
        for query in search_queries:
            print(f"\nSearch: \"{query}\"")
            retriever = store.as_retriever(search_kwargs={"k": 2})
            docs = retriever.invoke(query)
            
            if docs:
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content[:100].replace('\n', ' ')
                    print(f"  {i}. {content}...")
            else:
                print("  No relevant documents found")
        
        # Check data file status
        print("\n📁 ICD-11 Data File Check:")
        if os.path.exists("icd11_ch6_data"):
            csv_file = "icd11_ch6_data/icd11_ch6_entities.csv"
            raw_dir = "icd11_ch6_data/raw"
            
            if os.path.exists(csv_file):
                print(f"✅ CSV entity file exists")
            
            if os.path.exists(raw_dir):
                json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
                print(f"✅ JSON data files: {len(json_files)} files")
        else:
            print("⚠️  ICD-11 data directory does not exist")
        
        return True
        
    except Exception as e:
        print(f"❌ ICD-11 demo failed: {e}")
        return False

def demo_integrated_conversation():
    """Demo integrated conversation functionality"""
    print("\n" + "="*50)
    print("💬 CBT + ICD-11 Integrated Conversation Demo")
    print("="*50)
    
    try:
        from CBT_System.integration import CBTIntegration
        cbt = CBTIntegration()
        
        # Simulate conversation scenarios
        scenarios = [
            {
                "name": "Anxiety consultation",
                "input": "I've been feeling very anxious lately, can't sleep at night, heart racing",
                "description": "User describes anxiety symptoms, system should provide CBT anxiety coping techniques"
            },
            {
                "name": "Negative thinking",
                "input": "I always feel I can't do anything right, I'm a failure",
                "description": "Cognitive distortion, system should recommend cognitive restructuring techniques"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🎭 Scenario {i}: {scenario['name']}")
            print(f"📝 Description: {scenario['description']}")
            print(f"👤 User input: \"{scenario['input']}\"")
            
            # Check CBT relevance
            is_cbt_relevant = cbt.should_include_cbt(scenario['input'])
            print(f"🔍 CBT relevance: {'Yes' if is_cbt_relevant else 'No'}")
            
            if is_cbt_relevant:
                # Get CBT recommendations
                recommendations = cbt.cbt_kb.get_cbt_recommendation(scenario['input'])
                
                conditions = recommendations['query_analysis']['detected_conditions']
                techniques = recommendations['query_analysis']['suggested_techniques']
                
                print(f"🎯 Detected conditions: {', '.join(conditions) if conditions else 'General situation'}")
                print(f"🛠️  Recommended techniques: {', '.join(techniques) if techniques else 'General support'}")
                
                # Generate enhanced response
                if recommendations['recommended_techniques']:
                    context = f"User expressed: {scenario['input']}"
                    base_response = "I understand your feelings."
                    
                    enhanced_response = cbt.enhance_response_with_cbt(
                        user_query=scenario['input'],
                        context=context,
                        base_response=base_response
                    )
                    
                    print(f"🤖 AI response:")
                    print(f"   \"{enhanced_response[:150]}...\"")
            else:
                print("ℹ️  This input is not suitable for CBT technique enhancement")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated conversation demo failed: {e}")
        return False

def main():
    """Main demo function"""
    print("🌟 CBT + ICD-11 Mental Health Consultation System")
    print("   Quick Feature Demo")
    print("   " + "="*30)
    
    print("\n🎯 This demo will showcase:")
    print("  1. Core functionality of CBT cognitive behavioral therapy system")
    print("  2. ICD-11 medical knowledge retrieval functionality")
    print("  3. Integrated dialogue flow of both systems")
    
    print("\n⚠️  Please ensure:")
    print("  • CBT data has been collected and processed")
    print("  • ICD-11 vector index has been generated")
    print("  • Related Python dependencies are installed")
    
    input("\nPress Enter to start demo...")
    
    # Run various demos
    results = []
    
    # 1. CBT functionality demo
    cbt_success = demo_cbt_features()
    results.append(("CBT System", cbt_success))
    
    # 2. ICD-11 functionality demo
    icd11_success = demo_icd11_features()
    results.append(("ICD-11 System", icd11_success))
    
    # 3. Integrated conversation demo
    if cbt_success:  # Only run integrated demo when CBT is available
        dialogue_success = demo_integrated_conversation()
        results.append(("Integrated Conversation", dialogue_success))
    
    # Summarize results
    print("\n" + "="*50)
    print("📊 Demo Results Summary")
    print("="*50)
    
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\n🎯 Overall result: {success_count}/{total_count} modules working normally")
    
    if success_count == total_count:
        print("🎉 All functional modules are working properly!")
        print("💡 Suggestion: You can start the FastAPI server for complete experience")
        print("   Run: python fastapi_server.py")
    else:
        print("⚠️  Some functional modules need repair")
        print("💡 Suggestion: Check error messages from failed modules and fix related issues")
    
    print(f"\n📝 For detailed testing, please run: python test_cbt_icd11_demo.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred during demo: {e}")
        import traceback
        traceback.print_exc() 