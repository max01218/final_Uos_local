#!/usr/bin/env python3
"""
Test retrieval functionality specifically
"""

import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_retrieval():
    """Test the actual retrieval functionality"""
    print("=== Testing Retrieval Functionality ===")
    
    try:
        # Load embedding model
        print("Loading embedding model...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("✓ Embedding model loaded")
        
        # Load CBT index and metadata
        cbt_index_path = "CBT_System/cbt_data/embeddings/cbt_index.faiss"
        cbt_metadata_path = "CBT_System/cbt_data/embeddings/cbt_metadata.pkl"
        
        print(f"Loading CBT index from: {cbt_index_path}")
        cbt_index = faiss.read_index(cbt_index_path)
        print(f"✓ CBT index loaded: {cbt_index.ntotal} vectors, {cbt_index.d} dimensions")
        
        print(f"Loading CBT metadata from: {cbt_metadata_path}")
        with open(cbt_metadata_path, 'rb') as f:
            cbt_metadata = pickle.load(f)
        print(f"✓ CBT metadata loaded: {len(cbt_metadata)} items")
        
        # Test query
        query = "I feel persistent anxiety and insomnia. What can I do right now?"
        print(f"\nTesting query: '{query}'")
        
        # Generate embedding
        query_embedding = model.encode([query])
        print(f"✓ Query embedding generated: {query_embedding.shape}")
        
        # Search with different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            print(f"\n--- Testing threshold: {threshold} ---")
            
            # Search
            k = min(10, cbt_index.ntotal)
            distances, indices = cbt_index.search(query_embedding, k)
            
            # Filter by threshold
            valid_results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Convert distance to similarity score (1 - distance)
                similarity = 1 - distance
                if similarity >= threshold and idx < len(cbt_metadata):
                    valid_results.append((similarity, idx, cbt_metadata[idx]))
            
            print(f"Found {len(valid_results)} results above threshold {threshold}")
            
            for i, (similarity, idx, item) in enumerate(valid_results[:3]):
                print(f"  {i+1}. Similarity: {similarity:.3f}")
                print(f"     Type: {item.get('content_type', 'N/A')}")
                print(f"     Category: {item.get('primary_category', 'N/A')}")
                print(f"     Title: {item.get('title', 'N/A')[:80]}...")
                print()
        
        # Test with very low threshold to see all results
        print("--- All results (threshold 0.0) ---")
        distances, indices = cbt_index.search(query_embedding, min(5, cbt_index.ntotal))
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 - distance
            if idx < len(cbt_metadata):
                item = cbt_metadata[idx]
                print(f"  {i+1}. Similarity: {similarity:.3f}, Distance: {distance:.3f}")
                print(f"     Type: {item.get('content_type', 'N/A')}")
                print(f"     Title: {item.get('title', 'N/A')[:60]}...")
            else:
                print(f"  {i+1}. Similarity: {similarity:.3f}, Distance: {distance:.3f} (Index {idx} out of range)")
        
    except Exception as e:
        print(f"Error in retrieval test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()

