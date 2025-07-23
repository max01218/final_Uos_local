#!/usr/bin/env python3
"""
Detailed CBT Diagnostic Script
Deep dive into CBT search and recommendation issues
"""

import sys
import os
from pathlib import Path

# Add CBT_System to path
cbt_path = Path(__file__).parent / "CBT_System"
sys.path.append(str(cbt_path))

try:
    from integration import CBTIntegration
    
    def detailed_cbt_diagnosis():
        """Detailed CBT diagnosis"""
        
        print("Detailed CBT Diagnostic")
        print("=" * 50)
        print()
        
        # Initialize CBT integration
        cbt = CBTIntegration(base_dir="CBT_System/cbt_data")
        
        # Test problematic queries
        problem_queries = [
            "I'm struggling with anxiety and need help coping",
            "I feel anxious all the time, what can I do?"
        ]
        
        for query in problem_queries:
            print(f"Analyzing Query: '{query}'")
            print("-" * 50)
            
            # Step 1: Check relevance
            is_relevant = cbt.should_include_cbt(query)
            print(f"1. CBT Relevant: {is_relevant}")
            
            # Step 2: Analyze query
            analysis = cbt.cbt_kb.analyze_user_query(query)
            print(f"2. Query Analysis:")
            print(f"   Detected Conditions: {analysis['detected_conditions']}")
            print(f"   Suggested Techniques: {analysis['suggested_techniques']}")
            print(f"   Indicators: {analysis['indicators']}")
            
            # Step 3: Raw search results
            search_query = f"{query} {' '.join(analysis['indicators'])}"
            print(f"3. Search Query: '{search_query}'")
            
            raw_results = cbt.cbt_kb.search_cbt_techniques(search_query, top_k=5)
            print(f"4. Raw Search Results: {len(raw_results)} items")
            
            for i, result in enumerate(raw_results):
                print(f"   Result {i+1}:")
                print(f"     Type: {result.get('type', 'unknown')}")
                print(f"     Category: {result.get('category', 'unknown')}")
                print(f"     Score: {result.get('relevance_score', 0):.4f}")
                print(f"     Text: {result.get('text', 'No text')[:100]}...")
                print()
            
            # Step 4: Filtered results
            technique_results = [r for r in raw_results if r.get('type') == 'cbt_technique']
            content_results = [r for r in raw_results if r.get('type') == 'main_content']
            
            print(f"5. Filtered Results:")
            print(f"   CBT Techniques: {len(technique_results)}")
            print(f"   Main Content: {len(content_results)}")
            
            # Step 5: Get recommendations
            recommendations = cbt.cbt_kb.get_cbt_recommendation(query)
            print(f"6. Final Recommendations:")
            print(f"   Recommended Techniques: {len(recommendations['recommended_techniques'])}")
            print(f"   Supporting Content: {len(recommendations['supporting_content'])}")
            
            if recommendations['recommended_techniques']:
                for i, tech in enumerate(recommendations['recommended_techniques']):
                    print(f"   Technique {i+1}: {tech.get('category', 'unknown')} - {tech.get('text', 'No text')[:50]}...")
            
            print("\n" + "="*70 + "\n")
        
        # Test working queries for comparison
        print("COMPARISON - Working Queries:")
        print("=" * 50)
        
        working_queries = [
            "How can I stop negative thinking?",
            "Help me with panic attacks"
        ]
        
        for query in working_queries:
            print(f"Query: '{query}'")
            recommendations = cbt.cbt_kb.get_cbt_recommendation(query)
            print(f"Techniques Found: {len(recommendations['recommended_techniques'])}")
            
            if recommendations['recommended_techniques']:
                for tech in recommendations['recommended_techniques']:
                    print(f"  - {tech.get('category', 'unknown')}: {tech.get('text', 'No text')[:60]}...")
            print()
            
        # Check metadata distribution
        print("CBT Knowledge Base Analysis:")
        print("=" * 50)
        
        if cbt.cbt_kb.metadata:
            types = {}
            categories = {}
            
            for item in cbt.cbt_kb.metadata:
                item_type = item.get('type', 'unknown')
                item_category = item.get('category', 'unknown')
                
                types[item_type] = types.get(item_type, 0) + 1
                categories[item_category] = categories.get(item_category, 0) + 1
            
            print(f"Total Items: {len(cbt.cbt_kb.metadata)}")
            print(f"Types: {types}")
            print(f"Categories: {categories}")
            
            # Show sample CBT techniques
            cbt_techniques = [item for item in cbt.cbt_kb.metadata if item.get('type') == 'cbt_technique']
            print(f"\nSample CBT Techniques ({len(cbt_techniques)} total):")
            for i, tech in enumerate(cbt_techniques[:3]):
                print(f"  {i+1}. {tech.get('category', 'unknown')}: {tech.get('text', 'No text')[:100]}...")
    
    if __name__ == "__main__":
        detailed_cbt_diagnosis()
        
except ImportError as e:
    print(f"CBT Integration not available: {e}")
    sys.exit(1) 