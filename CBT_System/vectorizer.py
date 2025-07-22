#!/usr/bin/env python3
"""
CBT Vectorizer
Creates vector embeddings and FAISS index for CBT data
"""

import json
import os
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Any
from datetime import datetime

# Try to import required libraries
try:
    import faiss
except ImportError:
    print("FAISS not installed. Install with: pip install faiss-cpu")
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Sentence-transformers not installed. Install with: pip install sentence-transformers")
    SentenceTransformer = None

class CBTVectorizer:
    def __init__(self, base_dir="cbt_data", model_name="all-MiniLM-L6-v2"):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.setup_logging()
        
        # Initialize embedding model
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            
        # FAISS index
        self.index = None
        self.metadata = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_dir / 'vectorization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_processed_data(self) -> List[Dict]:
        """Load processed CBT data"""
        processed_data = []
        processed_dir = self.base_dir / "raw_data" / "processed"
        
        # Find the most recent processed file
        json_files = list(processed_dir.glob("cbt_processed_*.json"))
        if not json_files:
            self.logger.error("No processed data found")
            return []
            
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                
            self.logger.info(f"Loaded {len(processed_data)} processed items from {latest_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            
        return processed_data
        
    def prepare_text_chunks(self, data: List[Dict]) -> List[Dict]:
        """Prepare text chunks for embedding"""
        chunks = []
        
        for item in data:
            content = item.get('content', '')
            title = item.get('title', '')
            techniques = item.get('techniques', [])
            
            # Main content chunk
            main_text = f"{title}. {content}"
            if len(main_text) > 100:  # Only meaningful content
                chunks.append({
                    'text': main_text,
                    'type': 'main_content',
                    'source_url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'category': item.get('primary_category', 'general'),
                    'quality_score': item.get('quality_score', 0),
                    'metadata': {
                        'title': title,
                        'word_count': item.get('word_count', 0),
                        'techniques_count': len(techniques)
                    }
                })
                
            # Technique-specific chunks
            for technique in techniques:
                if technique.get('confidence', 0) > 0.3:  # High confidence techniques
                    technique_text = f"CBT Technique: {technique['technique']}. {technique['context']}"
                    
                    chunks.append({
                        'text': technique_text,
                        'type': 'cbt_technique',
                        'source_url': item.get('url', ''),
                        'source': item.get('source', ''),
                        'category': technique['category'],
                        'quality_score': technique.get('confidence', 0),
                        'metadata': {
                            'technique_name': technique['technique'],
                            'technique_category': technique['category'],
                            'confidence': technique.get('confidence', 0)
                        }
                    })
                    
        self.logger.info(f"Prepared {len(chunks)} text chunks for embedding")
        return chunks
        
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not self.embedding_model:
            self.logger.error("No embedding model available")
            return np.array([])
            
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            self.logger.info(f"Creating embeddings for {len(texts)} chunks")
            
            # Create embeddings in batches to manage memory
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
                embeddings.append(batch_embeddings)
                
                self.logger.info(f"Processed batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
                
            all_embeddings = np.vstack(embeddings)
            self.logger.info(f"Created embeddings shape: {all_embeddings.shape}")
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {e}")
            return np.array([])
            
    def create_faiss_index(self, embeddings: np.ndarray) -> bool:
        """Create FAISS index from embeddings"""
        if not faiss:
            self.logger.error("FAISS not available")
            return False
            
        if embeddings.size == 0:
            self.logger.error("No embeddings to index")
            return False
            
        try:
            # Create FAISS index
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings.astype('float32')
            faiss.normalize_L2(normalized_embeddings)
            
            # Add to index
            self.index.add(normalized_embeddings)
            
            self.logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {e}")
            return False
            
    def save_index_and_metadata(self, chunks: List[Dict]) -> bool:
        """Save FAISS index and metadata"""
        if not self.index:
            self.logger.error("No index to save")
            return False
            
        try:
            embeddings_dir = self.base_dir / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save FAISS index
            index_path = embeddings_dir / f"cbt_index_{timestamp}.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = embeddings_dir / f"cbt_metadata_{timestamp}.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(chunks, f)
                
            # Save current links (for easy loading)
            current_index_path = embeddings_dir / "cbt_index.faiss"
            current_metadata_path = embeddings_dir / "cbt_metadata.pkl"
            
            # Create symlinks or copy
            if current_index_path.exists():
                current_index_path.unlink()
            if current_metadata_path.exists():
                current_metadata_path.unlink()
                
            # Copy files
            import shutil
            shutil.copy2(index_path, current_index_path)
            shutil.copy2(metadata_path, current_metadata_path)
            
            # Save summary info
            summary = {
                'created_at': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'embedding_model': self.model_name,
                'embedding_dimension': self.embedding_dim,
                'index_path': str(index_path),
                'metadata_path': str(metadata_path),
                'categories': list(set(chunk['category'] for chunk in chunks)),
                'sources': list(set(chunk['source'] for chunk in chunks))
            }
            
            summary_path = embeddings_dir / "cbt_index_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved index and metadata to {embeddings_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
            
    def test_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Test search functionality"""
        if not self.index or not self.embedding_model:
            self.logger.error("Index or embedding model not available")
            return []
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['search_score'] = float(score)
                    result['search_rank'] = i + 1
                    results.append(result)
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Search test failed: {e}")
            return []
            
    def vectorize_all_data(self) -> bool:
        """Main vectorization pipeline"""
        self.logger.info("Starting CBT data vectorization")
        
        # Load processed data
        processed_data = self.load_processed_data()
        if not processed_data:
            self.logger.error("No processed data to vectorize")
            return False
            
        # Prepare text chunks
        chunks = self.prepare_text_chunks(processed_data)
        if not chunks:
            self.logger.error("No text chunks prepared")
            return False
            
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        if embeddings.size == 0:
            self.logger.error("Failed to create embeddings")
            return False
            
        # Create FAISS index
        if not self.create_faiss_index(embeddings):
            self.logger.error("Failed to create FAISS index")
            return False
            
        # Save everything
        self.metadata = chunks
        if not self.save_index_and_metadata(chunks):
            self.logger.error("Failed to save index and metadata")
            return False
            
        # Test search
        test_queries = [
            "cognitive behavioral therapy for depression",
            "anxiety coping strategies", 
            "thought challenging techniques"
        ]
        
        self.logger.info("Testing search functionality")
        for query in test_queries:
            results = self.test_search(query, top_k=3)
            self.logger.info(f"Query: '{query}' -> {len(results)} results")
            
        self.logger.info("Vectorization completed successfully")
        return True

def main():
    """Main vectorization function"""
    print("CBT Data Vectorization System")
    print("=" * 40)
    
    if not SentenceTransformer:
        print("ERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return False
        
    if not faiss:
        print("ERROR: FAISS not installed")
        print("Install with: pip install faiss-cpu")
        return False
        
    vectorizer = CBTVectorizer()
    
    try:
        success = vectorizer.vectorize_all_data()
        
        if success:
            print("\nVectorization completed successfully!")
            print("CBT knowledge base is ready for use")
        else:
            print("Vectorization failed")
            
        return success
        
    except Exception as e:
        print(f"Vectorization failed: {e}")
        return False

if __name__ == "__main__":
    main() 