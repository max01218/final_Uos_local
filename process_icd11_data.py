#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICD-11 Chapter 6 Data Processing Pipeline
Five stages: Data validation → Text preprocessing → Embedding & Indexing → Retrieval validation → RAG integration
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Stage 1: Data validation
def validate_data(csv_path: str) -> pd.DataFrame:
    """
    Stage 1: Data validation
    - Check CSV row count = 720 (including header)
    - Check missing fields (< 5%)
    - Remove duplicate codes
    """
    print("=" * 60)
    print("STAGE 1: DATA VALIDATION")
    print("=" * 60)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    
    # Check row count
    expected_rows = 720  # including header
    actual_rows = len(df) + 1  # +1 for header
    print(f"Row count check: {actual_rows} rows (expected: {expected_rows})")
    
    if actual_rows != expected_rows:
        print(f"⚠️  Warning: Row count mismatch! Expected {expected_rows}, got {actual_rows}")
    
    # Check missing fields
    missing_percentages = df.isna().mean() * 100
    print("\nMissing field percentages:")
    for col, pct in missing_percentages.items():
        print(f"  {col}: {pct:.2f}%")
    
    # Check for high missing rates
    high_missing = missing_percentages[missing_percentages > 5]
    if len(high_missing) > 0:
        print(f"⚠️  Warning: High missing rates in: {list(high_missing.index)}")
    
    # Remove duplicates
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['code'])
    final_count = len(df_clean)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate codes")
    
    print(f"Final clean data shape: {df_clean.shape}")
    return df_clean

# Stage 2: Text preprocessing
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Text preprocessing
    - Combine title/definition/inclusions/exclusions into one text
    - Clean extra whitespace and HTML
    """
    print("\n" + "=" * 60)
    print("STAGE 2: TEXT PREPROCESSING")
    print("=" * 60)
    
    def clean_text(text):
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text.strip()
    
    def combine_text_fields(row):
        """Combine all text fields into one comprehensive text"""
        parts = []
        
        # Add title
        if not pd.isna(row['title']) and row['title'] != "":
            parts.append(f"Title: {clean_text(row['title'])}")
        
        # Add definition
        if not pd.isna(row['definition']) and row['definition'] != "":
            parts.append(f"Definition: {clean_text(row['definition'])}")
        
        # Add inclusions
        if not pd.isna(row['inclusions']) and row['inclusions'] != "":
            parts.append(f"Inclusions: {clean_text(row['inclusions'])}")
        
        # Add exclusions
        if not pd.isna(row['exclusions']) and row['exclusions'] != "":
            parts.append(f"Exclusions: {clean_text(row['exclusions'])}")
        
        return " | ".join(parts)
    
    # Apply text cleaning
    print("Cleaning text fields...")
    for col in ['title', 'definition', 'inclusions', 'exclusions']:
        df[f'{col}_clean'] = df[col].apply(clean_text)
    
    # Combine all text fields
    print("Combining text fields...")
    df['combined_text'] = df.apply(combine_text_fields, axis=1)
    
    # Check text length distribution
    text_lengths = df['combined_text'].str.len()
    print(f"\nText length statistics:")
    print(f"  Mean: {text_lengths.mean():.1f} characters")
    print(f"  Median: {text_lengths.median():.1f} characters")
    print(f"  Min: {text_lengths.min()} characters")
    print(f"  Max: {text_lengths.max()} characters")
    
    # Show sample of combined text
    print(f"\nSample combined text:")
    sample_text = df['combined_text'].iloc[0]
    print(f"  {sample_text[:200]}...")
    
    return df

# Stage 3: Embedding and Indexing
def create_embeddings_and_index(df: pd.DataFrame) -> Tuple[List, object]:
    """
    Stage 3: Embedding and Indexing
    - Use sentence-transformers (all-MiniLM-L6-v2)
    - Generate 384-d vectors
    - Build FAISS index (IndexFlatIP)
    """
    print("\n" + "=" * 60)
    print("STAGE 3: EMBEDDING AND INDEXING")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f" Error: Missing required packages. Please install:")
        print(f"  pip install sentence-transformers faiss-cpu")
        return None, None
    
    # Load model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("Generating embeddings...")
    texts = df['combined_text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vector dimension: {embeddings.shape[1]}")
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    print(f"FAISS index built with {index.ntotal} vectors")
    
    return embeddings, index

# Stage 4: Retrieval validation
def validate_retrieval(df: pd.DataFrame, embeddings: List, index: object, model) -> float:
    """
    Stage 4: Retrieval validation
    - Random 20 test queries
    - Check Top-1 hit rate >= 0.8
    - Verify similarity score distribution
    """
    print("\n" + "=" * 60)
    print("STAGE 4: RETRIEVAL VALIDATION")
    print("=" * 60)
    
    # Generate test queries (using titles as queries)
    test_indices = np.random.choice(len(df), 20, replace=False)
    test_queries = df.iloc[test_indices]['title'].tolist()
    test_targets = test_indices.tolist()
    
    print(f"Generated {len(test_queries)} test queries")
    
    # Test retrieval
    correct_hits = 0
    similarity_scores = []
    
    for i, (query, target_idx) in enumerate(zip(test_queries, test_targets)):
        # Encode query
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), k=5)
        
        # Check if top result is correct
        if indices[0][0] == target_idx:
            correct_hits += 1
        
        similarity_scores.append(scores[0][0])
        
        print(f"  Query {i+1}: '{query[:50]}...'")
        print(f"    Top result: {indices[0][0]} (target: {target_idx})")
        print(f"    Similarity score: {scores[0][0]:.3f}")
    
    # Calculate hit rate
    hit_rate = correct_hits / len(test_queries)
    print(f"\nRetrieval Results:")
    print(f"  Correct hits: {correct_hits}/{len(test_queries)}")
    print(f"  Hit rate: {hit_rate:.3f} ({hit_rate*100:.1f}%)")
    
    # Check if hit rate meets requirement
    if hit_rate >= 0.8:
        print(f" Hit rate meets requirement (>= 0.8)")
    else:
        print(f" Hit rate below requirement (>= 0.8)")
    
    # Similarity score statistics
    print(f"\nSimilarity score statistics:")
    print(f"  Mean: {np.mean(similarity_scores):.3f}")
    print(f"  Std: {np.std(similarity_scores):.3f}")
    print(f"  Min: {np.min(similarity_scores):.3f}")
    print(f"  Max: {np.max(similarity_scores):.3f}")
    
    return hit_rate

# Stage 5: RAG Integration
def create_rag_system(df: pd.DataFrame, embeddings: List, index: object, model) -> Dict:
    """
    Stage 5: RAG Integration
    - Create RetrievalQA Chain
    - Custom prompt template
    - Prepare for Qwen-Chat API integration
    """
    print("\n" + "=" * 60)
    print("STAGE 5: RAG INTEGRATION")
    print("=" * 60)
    
    class ICD11RAGSystem:
        def __init__(self, df, embeddings, index, model):
            self.df = df
            self.embeddings = embeddings
            self.index = index
            self.model = model
        
        def retrieve(self, query: str, k: int = 3) -> List[Dict]:
            """Retrieve relevant documents for a query"""
            # Encode query
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k=k)
            
            # Return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.df):
                    row = self.df.iloc[idx]
                    results.append({
                        'rank': i + 1,
                        'code': row['code'],
                        'title': row['title'],
                        'definition': row['definition'],
                        'similarity_score': float(score),
                        'combined_text': row['combined_text']
                    })
            
            return results
        
        def generate_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
            """Generate prompt with retrieved context"""
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(f"Code: {doc['code']}\nTitle: {doc['title']}\nDefinition: {doc['definition']}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""You are a medical expert assistant. Based on the following ICD-11 mental health disorder information, please answer the user's question.

Context:
{context}

User Question: {query}

Please provide a comprehensive answer based on the provided context. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
            
            return prompt
        
        def query(self, question: str, k: int = 3) -> Dict:
            """Complete RAG query process"""
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(question, k)
            
            # Generate prompt
            prompt = self.generate_prompt(question, retrieved_docs)
            
            return {
                'question': question,
                'retrieved_documents': retrieved_docs,
                'prompt': prompt,
                'context_length': len(prompt)
            }
    
    # Create RAG system
    rag_system = ICD11RAGSystem(df, embeddings, index, model)
    
    # Test the system
    test_questions = [
        "What is schizophrenia?",
        "What are the symptoms of anxiety disorders?",
        "How is depression classified in ICD-11?",
        "What are neurodevelopmental disorders?",
        "What is the difference between mood disorders and anxiety disorders?"
    ]
    
    print("Testing RAG system with sample questions:")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = rag_system.query(question)
        
        print(f"Retrieved {len(result['retrieved_documents'])} documents:")
        for doc in result['retrieved_documents']:
            print(f"  - {doc['title']} (Code: {doc['code']}, Score: {doc['similarity_score']:.3f})")
        
        print(f"Generated prompt length: {result['context_length']} characters")
    
    return {
        'rag_system': rag_system,
        'df': df,
        'embeddings': embeddings,
        'index': index,
        'model': model
    }

# Main processing pipeline
def main():
    """Main processing pipeline"""
    print("ICD-11 Chapter 6 Data Processing Pipeline")
    print("=" * 60)
    
    # File paths
    csv_path = "icd11_ch6_data\icd11_ch6_entities.csv"
    
    # Check if files exist
    if not Path(csv_path).exists():
        print(f" Error: {csv_path} not found!")
        return
    
    try:
        # Stage 1: Data validation
        df = validate_data(csv_path)
        
        # Stage 2: Text preprocessing
        df_processed = preprocess_text(df)
        
        # Stage 3: Embedding and Indexing
        embeddings, index = create_embeddings_and_index(df_processed)
        
        if embeddings is None or index is None:
            print(" Failed to create embeddings and index")
            return
        
        # Load model for validation
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Stage 4: Retrieval validation
        hit_rate = validate_retrieval(df_processed, embeddings, index, model)
        
        # Stage 5: RAG Integration
        rag_results = create_rag_system(df_processed, embeddings, index, model)
        
        # Save processed data
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)
        
        # Save processed CSV
        output_csv = "icd11_ch6_processed.csv"
        df_processed.to_csv(output_csv, index=False)
        print(f"Saved processed CSV: {output_csv}")
        
        # Save embeddings
        import pickle
        with open('icd11_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        print(" Saved embeddings: icd11_embeddings.pkl")
        
        # Save FAISS index
        import faiss
        faiss.write_index(index, 'icd11_index.faiss')
        print(" Saved FAISS index: icd11_index.faiss")
        
        # Save RAG system
        with open('icd11_rag_system.pkl', 'wb') as f:
            pickle.dump(rag_results, f)
        print(" Saved RAG system: icd11_rag_system.pkl")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final statistics:")
        print(f"  - Processed {len(df_processed)} ICD-11 disorders")
        print(f"  - Generated {embeddings.shape[1]}-dimensional embeddings")
        print(f"  - Retrieval hit rate: {hit_rate:.3f}")
        print(f"  - RAG system ready for Qwen-Chat API integration")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 