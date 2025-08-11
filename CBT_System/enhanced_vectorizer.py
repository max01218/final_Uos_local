#!/usr/bin/env python3
"""
Enhanced CBT Vectorization System
Multi-level vectorization with quality weighting and hierarchical indexing
"""

import json
import os
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch

class EnhancedCBTVectorizer:
    def __init__(self, base_dir="cbt_data"):
        # Resolve base_dir relative to this file if not absolute
        base_dir_path = Path(base_dir)
        if not base_dir_path.is_absolute():
            base_dir_path = Path(__file__).resolve().parent / base_dir_path
        # Ensure directory exists
        base_dir_path.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir_path
        self.setup_logging()
        
        # Model configurations
        self.embedding_models = {
            "primary": "sentence-transformers/all-mpnet-base-v2",  # Unified embedding model
            "clinical": "sentence-transformers/all-mpnet-base-v2",  # Better for clinical text
            "domain_specific": "sentence-transformers/all-roberta-large-v1"  # Best quality
        }
        
        # Vectorization strategies
        self.vectorization_strategies = {
            "content_based": "full_content",
            "technique_focused": "technique_extraction",
            "step_by_step": "structured_steps",
            "concept_based": "key_concepts"
        }
        
        # Quality score thresholds for different index levels
        self.quality_tiers = {
            "premium": 85,    # Highest quality content
            "standard": 70,   # Good quality content
            "basic": 60       # Acceptable quality content
        }
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
    def setup_logging(self):
        """Setup enhanced logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(log_format)
        
        # Ensure base dir exists before creating log file
        self.base_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.base_dir / 'enhanced_vectorization.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        
        self.logger = logging.getLogger("EnhancedCBTVectorizer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_enhanced_processed_data(self) -> Optional[Tuple[List[Dict], Dict]]:
        """Load the most recent enhanced processed data"""
        processed_dir = self.base_dir / "raw_data" / "processed"
        
        if not processed_dir.exists():
            self.logger.error("Processed data directory not found")
            return None
            
        # Find the most recent enhanced processed file
        enhanced_files = list(processed_dir.glob("cbt_enhanced_processed_*.json"))
        
        if not enhanced_files:
            self.logger.error("No enhanced processed data files found")
            return None
            
        # Sort by modification time and get the most recent
        latest_file = max(enhanced_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            metadata = data.get("metadata", {})
            items = data.get("data", [])
            
            self.logger.info(f"Loaded {len(items)} enhanced processed items from {latest_file}")
            return items, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load enhanced processed data: {e}")
            return None
            
    def load_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Load and configure embedding model"""
        try:
            model = SentenceTransformer(model_name, device=self.device)
            self.logger.info(f"Loaded embedding model: {model_name} on {self.device}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to default model
            return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=self.device)
            
    def extract_vectorization_text(self, item: Dict, strategy: str) -> str:
        """Extract appropriate text for vectorization based on strategy"""
        
        if strategy == "full_content":
            return item.get("content", "")
            
        elif strategy == "technique_extraction":
            # Focus on technique-specific content
            classification = item.get("classification", {})
            structured_info = item.get("structured_information", {})
            
            text_parts = []
            
            # Add primary category information
            primary_category = classification.get("primary_category", "")
            if primary_category:
                text_parts.append(f"CBT technique category: {primary_category}")
                
            # Add matched keywords
            category_scores = classification.get("extracted_features", {}).get("technique_category_scores", {})
            for category, data in category_scores.items():
                if data.get("matched_keywords"):
                    keywords = " ".join(data["matched_keywords"])
                    text_parts.append(f"{category} keywords: {keywords}")
                    
            # Add structured techniques
            techniques = structured_info.get("techniques", [])
            text_parts.extend(techniques[:3])  # Limit to top 3
            
            # Fallback to main content if no specific techniques found
            if not text_parts:
                text_parts.append(item.get("content", ""))
                
            return " ".join(text_parts)
            
        elif strategy == "structured_steps":
            # Focus on step-by-step content
            structured_info = item.get("structured_information", {})
            steps = structured_info.get("steps", [])
            
            if steps:
                step_texts = [f"Step {step['number']}: {step['description']}" for step in steps[:5]]
                return " ".join(step_texts)
            else:
                # Fallback to content
                return item.get("content", "")
                
        elif strategy == "key_concepts":
            # Focus on key concepts and points
            structured_info = item.get("structured_information", {})
            
            text_parts = []
            
            # Add key points
            key_points = structured_info.get("key_points", [])
            text_parts.extend(key_points[:5])
            
            # Add examples
            examples = structured_info.get("examples", [])
            text_parts.extend(examples[:3])
            
            # Add title if available
            title = item.get("title", "")
            if title:
                text_parts.insert(0, title)
                
            if not text_parts:
                text_parts.append(item.get("content", ""))
                
            return " ".join(text_parts)
            
        else:
            return item.get("content", "")
            
    def create_embeddings(self, items: List[Dict], model: SentenceTransformer, strategy: str) -> Tuple[np.ndarray, List[Dict]]:
        """Create embeddings with enhanced metadata"""
        
        texts = []
        metadata = []
        
        self.logger.info(f"Creating embeddings using strategy: {strategy}")
        
        for i, item in enumerate(items):
            # Extract text based on strategy
            text = self.extract_vectorization_text(item, strategy)
            
            if len(text.strip()) < 50:  # Skip very short texts
                continue
                
            texts.append(text)
            
            # Create enhanced metadata
            item_metadata = {
                "index": i,
                "source": item.get("source", "unknown"),
                "source_type": item.get("source_type", "unknown"),
                "quality_score": item.get("enhanced_quality_score", 0),
                "primary_category": item.get("classification", {}).get("primary_category"),
                "content_type": item.get("classification", {}).get("content_type"),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content_length": len(item.get("content", "")),
                "processing_timestamp": item.get("processing_timestamp", ""),
                "vectorization_strategy": strategy,
                "text_used_for_embedding": text[:200] + "..." if len(text) > 200 else text
            }
            
            metadata.append(item_metadata)
            
        if not texts:
            self.logger.warning("No valid texts found for embedding")
            return np.array([]), []
            
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_tensor=False,
                show_progress_bar=True if i == 0 else False
            )
            all_embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(all_embeddings)
        
        self.logger.info(f"Created {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        return embeddings, metadata
        
    def create_hierarchical_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict[str, faiss.IndexFlatIP]:
        """Create hierarchical FAISS indices based on quality tiers"""
        
        indices = {}
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        
        # Create indices for each quality tier
        for tier_name, min_score in self.quality_tiers.items():
            # Filter embeddings by quality score
            tier_mask = np.array([
                meta["quality_score"] >= min_score 
                for meta in metadata
            ])
            
            if np.sum(tier_mask) == 0:
                self.logger.warning(f"No items found for quality tier: {tier_name}")
                continue
                
            tier_embeddings = embeddings[tier_mask]
            tier_metadata = [meta for i, meta in enumerate(metadata) if tier_mask[i]]
            
            # Create FAISS index
            dimension = tier_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(tier_embeddings.astype('float32'))
            
            indices[tier_name] = {
                "index": index,
                "metadata": tier_metadata,
                "embedding_count": len(tier_embeddings)
            }
            
            self.logger.info(f"Created {tier_name} tier index with {len(tier_embeddings)} embeddings")
            
        # Create category-specific indices
        category_indices = self.create_category_indices(embeddings, metadata)
        indices.update(category_indices)
        
        return indices
        
    def create_category_indices(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict[str, Dict]:
        """Create category-specific indices"""
        
        category_indices = {}
        
        # Group by primary category
        category_groups = {}
        for i, meta in enumerate(metadata):
            category = meta.get("primary_category")
            if category:
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(i)
                
        # Create index for each category with sufficient items
        for category, indices_list in category_groups.items():
            if len(indices_list) >= 3:  # Minimum items for category index
                category_embeddings = embeddings[indices_list]
                category_metadata = [metadata[i] for i in indices_list]
                
                # Normalize embeddings
                faiss.normalize_L2(category_embeddings.astype('float32'))
                
                # Create FAISS index
                dimension = category_embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                index.add(category_embeddings.astype('float32'))
                
                category_indices[f"category_{category}"] = {
                    "index": index,
                    "metadata": category_metadata,
                    "embedding_count": len(category_embeddings),
                    "category": category
                }
                
                self.logger.info(f"Created category index for {category}: {len(category_embeddings)} embeddings")
                
        return category_indices
        
    def optimize_index_performance(self, index: faiss.IndexFlatIP, embedding_count: int) -> faiss.Index:
        """Optimize index for better search performance"""
        
        if embedding_count < 100:
            # For small datasets, flat index is sufficient
            return index
            
        elif embedding_count < 1000:
            # Use IVF for medium datasets
            quantizer = faiss.IndexFlatIP(index.d)
            nlist = min(100, embedding_count // 10)
            optimized_index = faiss.IndexIVFFlat(quantizer, index.d, nlist)
            
            # Train and add vectors
            vectors = index.reconstruct_n(0, index.ntotal)
            optimized_index.train(vectors)
            optimized_index.add(vectors)
            
            return optimized_index
            
        else:
            # Use IVF with PQ for large datasets
            quantizer = faiss.IndexFlatIP(index.d)
            nlist = min(1024, embedding_count // 50)
            m = 8  # Number of subquantizers
            optimized_index = faiss.IndexIVFPQ(quantizer, index.d, nlist, m, 8)
            
            # Train and add vectors
            vectors = index.reconstruct_n(0, index.ntotal)
            optimized_index.train(vectors)
            optimized_index.add(vectors)
            
            return optimized_index
            
    def save_enhanced_vectorization(self, indices: Dict, model_name: str, strategy: str, processing_metadata: Dict) -> str:
        """Save enhanced vectorization results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / "embeddings"
        
        # Create summary information
        summary = {
            "creation_timestamp": datetime.now().isoformat(),
            "embedding_model": model_name,
            "vectorization_strategy": strategy,
            "device_used": self.device,
            "processing_metadata": processing_metadata,
            "indices_summary": {}
        }
        
        saved_files = {}
        
        # Save each index separately
        for index_name, index_data in indices.items():
            index_obj = index_data["index"]
            metadata = index_data["metadata"]
            
            # Optimize index if needed
            optimized_index = self.optimize_index_performance(index_obj, index_data["embedding_count"])
            
            # Save FAISS index
            index_filename = f"cbt_index_{index_name}_{timestamp}.faiss"
            index_path = output_dir / index_filename
            faiss.write_index(optimized_index, str(index_path))
            
            # Save metadata
            metadata_filename = f"cbt_metadata_{index_name}_{timestamp}.pkl"
            metadata_path = output_dir / metadata_filename
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            # Update summary
            summary["indices_summary"][index_name] = {
                "index_file": index_filename,
                "metadata_file": metadata_filename,
                "embedding_count": index_data["embedding_count"],
                "index_type": type(optimized_index).__name__,
                "dimension": optimized_index.d if hasattr(optimized_index, 'd') else "unknown"
            }
            
            saved_files[index_name] = {
                "index_path": str(index_path),
                "metadata_path": str(metadata_path)
            }
            
        # Create main index (highest quality tier) for backward compatibility
        if "premium" in indices:
            main_index = indices["premium"]["index"]
            main_metadata = indices["premium"]["metadata"]
        elif "standard" in indices:
            main_index = indices["standard"]["index"]
            main_metadata = indices["standard"]["metadata"]
        else:
            # Use the first available index
            first_key = list(indices.keys())[0]
            main_index = indices[first_key]["index"]
            main_metadata = indices[first_key]["metadata"]
            
        # Save main index as default
        main_index_path = output_dir / "cbt_index.faiss"
        main_metadata_path = output_dir / "cbt_metadata.pkl"
        
        faiss.write_index(main_index, str(main_index_path))
        with open(main_metadata_path, 'wb') as f:
            pickle.dump(main_metadata, f)
            
        # Save summary
        summary_path = output_dir / f"cbt_index_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # Update main summary file
        main_summary_path = output_dir / "cbt_index_summary.json"
        with open(main_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Enhanced vectorization saved to {output_dir}")
        self.logger.info(f"Total indices created: {len(indices)}")
        
        return str(summary_path)
        
    def test_search_functionality(self, indices: Dict, model: SentenceTransformer) -> Dict:
        """Test search functionality across all indices"""
        
        test_queries = [
            "cognitive behavioral therapy for depression",
            "anxiety coping strategies", 
            "thought challenging techniques",
            "mindfulness meditation exercises",
            "problem solving therapy steps"
        ]
        
        test_results = {}
        
        for query in test_queries:
            query_embedding = model.encode([query])
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            query_results = {}
            
            for index_name, index_data in indices.items():
                try:
                    index_obj = index_data["index"]
                    metadata = index_data["metadata"]
                    
                    # Search
                    scores, indices_found = index_obj.search(query_embedding.astype('float32'), min(3, len(metadata)))
                    
                    results = []
                    for i, (score, idx) in enumerate(zip(scores[0], indices_found[0])):
                        if idx >= 0 and idx < len(metadata):
                            result = {
                                "score": float(score),
                                "metadata": metadata[idx]
                            }
                            results.append(result)
                            
                    query_results[index_name] = results
                    
                except Exception as e:
                    self.logger.error(f"Search test failed for {index_name} with query '{query}': {e}")
                    query_results[index_name] = []
                    
            test_results[query] = query_results
            
        return test_results
        
    def vectorize_enhanced_data(self, model_choice: str = "primary", strategy: str = "content_based") -> Optional[str]:
        """Main method to vectorize enhanced processed data"""
        
        self.logger.info("Starting enhanced vectorization process")
        
        # Load processed data
        data_result = self.load_enhanced_processed_data()
        if not data_result:
            return None
            
        items, processing_metadata = data_result
        
        # Load embedding model
        model_name = self.embedding_models[model_choice]
        model = self.load_embedding_model(model_name)
        
        # Create embeddings
        embeddings, metadata = self.create_embeddings(items, model, strategy)
        
        if embeddings.size == 0:
            self.logger.error("No embeddings created")
            return None
            
        # Create hierarchical indices
        indices = self.create_hierarchical_index(embeddings, metadata)
        
        if not indices:
            self.logger.error("No indices created")
            return None
            
        # Test search functionality
        test_results = self.test_search_functionality(indices, model)
        
        # Save everything
        result_path = self.save_enhanced_vectorization(
            indices, model_name, strategy, processing_metadata
        )
        
        self.logger.info("Enhanced vectorization completed successfully")
        return result_path

if __name__ == "__main__":
    vectorizer = EnhancedCBTVectorizer()
    
    print("Starting enhanced CBT vectorization...")
    
    # Try different model and strategy combinations
    combinations = [
        ("primary", "content_based"),
        ("clinical", "technique_focused"), 
        ("primary", "structured_steps")
    ]
    
    for model_choice, strategy in combinations:
        print(f"\nVectorizing with model: {model_choice}, strategy: {strategy}")
        result = vectorizer.vectorize_enhanced_data(model_choice, strategy)
        
        if result:
            print(f"Vectorization successful: {result}")
        else:
            print("Vectorization failed")
            
    print("\nEnhanced vectorization process completed") 