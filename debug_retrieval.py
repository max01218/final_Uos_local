#!/usr/bin/env python3
"""
Debug script to test knowledge base retrieval functionality
"""

import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_cbt_metadata():
    """Test CBT metadata loading and statistics"""
    print("=== Testing CBT Metadata ===")
    
    cbt_metadata_path = "CBT_System/cbt_data/embeddings/cbt_metadata.pkl"
    if Path(cbt_metadata_path).exists():
        with open(cbt_metadata_path, 'rb') as f:
            cbt_metadata = pickle.load(f)
        
        print(f"CBT metadata loaded: {len(cbt_metadata)} items")
        
        # Count techniques and content
        cbt_techniques = 0
        cbt_content = 0
        
        for item in cbt_metadata:
            if isinstance(item, dict):
                content_type = (item.get('content_type') or '').lower()
                primary_category = (item.get('primary_category') or '').lower()
                
                # Count as technique if it's a technique, protocol, or assessment tool
                if any(keyword in content_type for keyword in ['technique', 'protocol', 'assessment']):
                    cbt_techniques += 1
                elif any(keyword in primary_category for keyword in ['technique', 'protocol', 'assessment']):
                    cbt_techniques += 1
                else:
                    cbt_content += 1
            else:
                cbt_content += 1
        
        print(f"CBT Statistics: {cbt_techniques} techniques, {cbt_content} content items")
        
        # Show first few items
        print("\nFirst 3 CBT items:")
        for i, item in enumerate(cbt_metadata[:3]):
            print(f"  {i+1}. Type: {item.get('content_type', 'N/A')}, Category: {item.get('primary_category', 'N/A')}")
    else:
        print(f"CBT metadata file not found: {cbt_metadata_path}")

def test_icd11_metadata():
    """Test ICD-11 metadata loading"""
    print("\n=== Testing ICD-11 Metadata ===")
    
    icd11_metadata_path = "embeddings/index.pkl"
    if Path(icd11_metadata_path).exists():
        with open(icd11_metadata_path, 'rb') as f:
            icd11_metadata = pickle.load(f)
        
        print(f"ICD-11 metadata loaded: {len(icd11_metadata) if isinstance(icd11_metadata, (list, tuple)) else 'Unknown'} items")
        
        if isinstance(icd11_metadata, tuple) and len(icd11_metadata) == 2:
            docstore, index_to_docstore_id = icd11_metadata
            print(f"  - Docstore type: {type(docstore)}")
            print(f"  - Index to docstore ID length: {len(index_to_docstore_id)}")
            
            if hasattr(docstore, '_dict'):
                print(f"  - Docstore dict items: {len(docstore._dict)}")
            elif hasattr(docstore, 'dict'):
                print(f"  - Docstore dict items: {len(docstore.dict)}")
    else:
        print(f"ICD-11 metadata file not found: {icd11_metadata_path}")

def test_embedding_model():
    """Test embedding model loading"""
    print("\n=== Testing Embedding Model ===")
    
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("Embedding model loaded successfully")
        
        # Test embedding
        test_text = "I feel persistent anxiety and insomnia"
        embedding = model.encode([test_text])
        print(f"Test embedding shape: {embedding.shape}")
        print(f"Test embedding sample: {embedding[0][:5]}...")
        
    except Exception as e:
        print(f"Error loading embedding model: {e}")

def test_faiss_indexes():
    """Test FAISS index loading"""
    print("\n=== Testing FAISS Indexes ===")
    
    # Test CBT index
    cbt_index_path = "CBT_System/cbt_data/embeddings/cbt_index.faiss"
    if Path(cbt_index_path).exists():
        try:
            cbt_index = faiss.read_index(cbt_index_path)
            print(f"CBT index loaded: {cbt_index.ntotal} vectors, {cbt_index.d} dimensions")
        except Exception as e:
            print(f"Error loading CBT index: {e}")
    else:
        print(f"CBT index file not found: {cbt_index_path}")
    
    # Test ICD-11 index
    icd11_index_path = "embeddings/index.faiss"
    if Path(icd11_index_path).exists():
        try:
            icd11_index = faiss.read_index(icd11_index_path)
            print(f"ICD-11 index loaded: {icd11_index.ntotal} vectors, {icd11_index.d} dimensions")
        except Exception as e:
            print(f"Error loading ICD-11 index: {e}")
    else:
        print(f"ICD-11 index file not found: {icd11_index_path}")

def test_simple_retrieval():
    """Test simple retrieval functionality"""
    print("\n=== Testing Simple Retrieval ===")
    
    try:
        # Load embedding model
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Load CBT index and metadata
        cbt_index_path = "CBT_System/cbt_data/embeddings/cbt_index.faiss"
        cbt_metadata_path = "CBT_System/cbt_data/embeddings/cbt_metadata.pkl"
        
        if Path(cbt_index_path).exists() and Path(cbt_metadata_path).exists():
            cbt_index = faiss.read_index(cbt_index_path)
            with open(cbt_metadata_path, 'rb') as f:
                cbt_metadata = pickle.load(f)
            
            # Test query
            query = "I feel persistent anxiety and insomnia. What can I do right now?"
            query_embedding = model.encode([query])
            
            # Search
            k = min(5, cbt_index.ntotal)
            distances, indices = cbt_index.search(query_embedding, k)
            
            print(f"Retrieval results for query: '{query}'")
            print(f"Found {len(indices[0])} results")
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(cbt_metadata):
                    item = cbt_metadata[idx]
                    print(f"  {i+1}. Distance: {distance:.3f}, Type: {item.get('content_type', 'N/A')}")
                    print(f"     Title: {item.get('title', 'N/A')[:100]}...")
                else:
                    print(f"  {i+1}. Distance: {distance:.3f}, Index {idx} out of range")
        else:
            print("CBT index or metadata not found")
            
    except Exception as e:
        print(f"Error in simple retrieval test: {e}")

if __name__ == "__main__":
    test_cbt_metadata()
    test_icd11_metadata()
    test_embedding_model()
    test_faiss_indexes()
    test_simple_retrieval()
