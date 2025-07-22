#!/usr/bin/env python3
"""
Fixed RAG Chain implementation with stable imports
"""

import os
import sys
from pathlib import Path

# Set environment variables to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Try to import with specific error handling
try:
    # First, try to import sentence_transformers directly
    import sentence_transformers
    print(f"sentence-transformers version: {sentence_transformers.__version__}")
    
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("Successfully imported all required packages")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying to fix sentence-transformers installation...")
    
    # Try alternative import approach
    try:
        # Force reload of sentence_transformers
        if 'sentence_transformers' in sys.modules:
            del sys.modules['sentence_transformers']
        
        import sentence_transformers
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Successfully imported using reload method")
        
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("Please run: pip install --upgrade sentence-transformers")
        sys.exit(1)

def load_embeddings():
    """Load embeddings with comprehensive error handling"""
    try:
        print("Loading HuggingFace embeddings...")
        
        # Try with specific model configuration
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Test the embedder
        test_text = "test"
        test_embedding = embedder.embed_query(test_text)
        print(f"Embedding test successful, dimension: {len(test_embedding)}")
        
        print("Embeddings loaded successfully")
        return embedder
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Trying alternative model...")
        
        try:
            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Test the embedder
            test_text = "test"
            test_embedding = embedder.embed_query(test_text)
            print(f"Alternative embedding test successful, dimension: {len(test_embedding)}")
            
            print("Alternative embeddings loaded successfully")
            return embedder
            
        except Exception as e2:
            print(f"Alternative embeddings also failed: {e2}")
            print("Trying third alternative...")
            
            try:
                embedder = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={"device": "cpu"}
                )
                print("Third alternative embeddings loaded successfully")
                return embedder
                
            except Exception as e3:
                print(f"All embedding models failed: {e3}")
                return None

def load_faiss_index(embedder):
    """Load FAISS index with error handling"""
    try:
        print("Loading FAISS index...")
        
        embeddings_dir = Path("embeddings")
        if not embeddings_dir.exists():
            print("embeddings directory not found!")
            print("Please run test.py first to generate embeddings")
            return None
        
        # Check what files are available
        files = list(embeddings_dir.glob("*"))
        print(f"Found {len(files)} files in embeddings directory:")
        for f in files:
            print(f"  - {f.name}")
        
        store = FAISS.load_local("embeddings", embedder,
                                 allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully")
        return store
        
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def load_llm():
    """Load LLM with error handling"""
    try:
        print("Loading Qwen model...")
        
        # Try with specific configuration
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            trust_remote_code=True,
            padding_side="left"
        )
        
        llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto"
        )
        
        print("Qwen model loaded successfully")
        return llm
        
    except Exception as e:
        print(f"Error loading Qwen model: {e}")
        print("This might be due to model size or memory constraints")
        print("Trying with CPU only...")
        
        try:
            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-1_8B-Chat",
                trust_remote_code=True
            )
            
            llm = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-1_8B-Chat",
                trust_remote_code=True,
                device_map="cpu"
            )
            
            print("Qwen model loaded successfully on CPU")
            return llm
            
        except Exception as e2:
            print(f"CPU loading also failed: {e2}")
            return None

def create_rag_chain(store, llm):
    """Create RAG chain with error handling"""
    try:
        print("Creating RAG chain...")
        
        # Create retriever
        retriever = store.as_retriever(search_kwargs={"k": 3})
        
        # Create prompt template
        PROMPT = PromptTemplate(
            template=(
                "You are a professional mental-health assistant.\n\n"
                "Context:\n{context}\n\n"
                "User Question: {question}\n\n"
                "Answer concisely and factually. "
                "Only use the given context; do not invent information."
            ),
            input_variables=["context", "question"],
        )
        
        # Create RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
        )
        
        print("RAG chain created successfully")
        return rag_chain
        
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        return None

def test_rag_system(rag_chain):
    """Test the RAG system with sample questions"""
    if rag_chain is None:
        print("RAG chain is not available for testing")
        return
    
    test_questions = [
        "What is depression?",
        "How is schizophrenia defined?",
        "What are the symptoms of anxiety disorder?"
    ]
    
    print("\n" + "="*60)
    print("Testing RAG System")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        try:
            answer = rag_chain({"query": question})
            print(f"Answer: {answer['result']}")
        except Exception as e:
            print(f"Error getting answer: {e}")
        
        print("\n" + "="*60)

def main():
    """Main function with comprehensive error handling"""
    print("Starting RAG Chain setup...")
    
    # Step 1: Load embeddings
    embedder = load_embeddings()
    if embedder is None:
        print("Failed to load embeddings. Exiting.")
        return
    
    # Step 2: Load FAISS index
    store = load_faiss_index(embedder)
    if store is None:
        print("Failed to load FAISS index. Exiting.")
        return
    
    # Step 3: Load LLM
    llm = load_llm()
    if llm is None:
        print("Failed to load LLM. Exiting.")
        return
    
    # Step 4: Create RAG chain
    rag_chain = create_rag_chain(store, llm)
    if rag_chain is None:
        print("Failed to create RAG chain. Exiting.")
        return
    
    # Step 5: Test the system
    test_rag_system(rag_chain)
    
    print("\nRAG Chain setup completed successfully!")

if __name__ == "__main__":
    main() 