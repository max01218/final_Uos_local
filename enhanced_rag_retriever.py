#!/usr/bin/env python3
"""
Enhanced RAG Retrieval System
Advanced retrieval with multi-level semantic similarity, intent recognition, and context awareness
"""

import json
import logging
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")

class QueryIntent(Enum):
    """User query intent categories"""
    INFORMATION_SEEKING = "information_seeking"
    EMOTIONAL_SUPPORT = "emotional_support"
    CRISIS_INTERVENTION = "crisis_intervention"
    SKILL_LEARNING = "skill_learning"
    SYMPTOM_ASSESSMENT = "symptom_assessment"
    TREATMENT_GUIDANCE = "treatment_guidance"
    GENERAL_CHAT = "general_chat"

class RetrievalStrategy(Enum):
    """Different retrieval strategies"""
    SEMANTIC_ONLY = "semantic_only"
    HYBRID_SEMANTIC_KEYWORD = "hybrid"
    CONTEXT_AWARE = "context_aware"
    INTENT_BASED = "intent_based"
    MULTI_MODAL_FUSION = "multi_modal_fusion"

@dataclass
class RetrievalContext:
    """Context for retrieval operations"""
    conversation_history: List[Dict] = None
    user_preferences: Dict = None
    session_topics: List[str] = None
    emotional_state: str = None
    urgency_level: int = 1  # 1-5 scale
    domain_focus: str = None  # 'cbt', 'icd11', 'both'

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata"""
    content: str
    source: str
    relevance_score: float
    confidence: float
    intent_match: float
    context_relevance: float
    metadata: Dict
    reasoning: str

class MentalHealthTerminology:
    """Mental health terminology and synonym management"""
    
    def __init__(self):
        self.symptom_synonyms = {
            "anxiety": ["anxious", "worried", "nervous", "panic", "fear", "stress", "tension"],
            "depression": ["depressed", "sad", "down", "low", "hopeless", "empty", "mood"],
            "stress": ["stressed", "overwhelmed", "pressure", "burden", "strain"],
            "anger": ["angry", "mad", "furious", "irritated", "frustrated", "rage"],
            "sleep": ["insomnia", "sleepless", "tired", "fatigue", "exhausted"],
            "thoughts": ["thinking", "mind", "ideas", "cognition", "mental"],
            "behavior": ["actions", "habits", "patterns", "conduct", "response"]
        }
        
        self.technique_synonyms = {
            "cognitive_restructuring": ["thought challenging", "cognitive reframing", "thinking patterns"],
            "relaxation": ["breathing", "meditation", "mindfulness", "calm", "peace"],
            "exposure": ["facing fears", "gradual exposure", "desensitization"],
            "behavioral_activation": ["activity scheduling", "engagement", "pleasant activities"]
        }
        
        self.urgency_indicators = {
            "high": ["suicide", "kill myself", "hurt myself", "can't go on", "end it all", "crisis"],
            "medium": ["desperate", "can't cope", "overwhelmed", "breaking point"],
            "low": ["struggling", "difficult", "hard time", "need help"]
        }
        
    def expand_query_terms(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Add synonyms for detected terms
        for main_term, synonyms in self.symptom_synonyms.items():
            if main_term in query_lower or any(syn in query_lower for syn in synonyms):
                expanded_terms.extend(synonyms)
                
        return list(set(expanded_terms))
        
    def detect_urgency_level(self, query: str) -> int:
        """Detect urgency level from query content"""
        query_lower = query.lower()
        
        for level, indicators in self.urgency_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                if level == "high":
                    return 5
                elif level == "medium":
                    return 3
                elif level == "low":
                    return 2
                    
        return 1  # Default low urgency

class IntentClassifier:
    """Classify user query intent"""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.CRISIS_INTERVENTION: [
                r"suicide|kill myself|hurt myself|end it all|can't go on",
                r"crisis|emergency|urgent|desperate|breaking point"
            ],
            QueryIntent.EMOTIONAL_SUPPORT: [
                r"feeling|feel|emotions|emotional|mood|upset|sad|lonely",
                r"need support|comfort|understand|listen|talk"
            ],
            QueryIntent.SKILL_LEARNING: [
                r"how to|teach me|learn|technique|strategy|skill|method",
                r"steps|guide|instruction|practice|exercise"
            ],
            QueryIntent.SYMPTOM_ASSESSMENT: [
                r"symptoms|signs|diagnosis|assess|evaluate|check",
                r"what is wrong|what's happening|am i|do i have"
            ],
            QueryIntent.TREATMENT_GUIDANCE: [
                r"treatment|therapy|medication|help|cure|fix",
                r"what should i do|recommendations|advice|guidance"
            ],
            QueryIntent.INFORMATION_SEEKING: [
                r"what is|tell me about|explain|information|definition",
                r"research|studies|facts|statistics|causes"
            ],
            QueryIntent.GENERAL_CHAT: [
                r"hello|hi|how are you|good morning|thank you",
                r"bye|goodbye|see you|take care"
            ]
        }
        
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
                
            if score > 0:
                intent_scores[intent] = score
                
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1] / len(query.split()) * 2, 1.0)
            return best_intent[0], confidence
        else:
            return QueryIntent.GENERAL_CHAT, 0.1

class SemanticSimilarityEngine:
    """Advanced semantic similarity computation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except:
            self.model = None
            
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def compute_semantic_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute semantic similarity scores"""
        if not self.model or not documents:
            return np.zeros(len(documents))
            
        try:
            query_embedding = self.model.encode([query])
            doc_embeddings = self.model.encode(documents)
            
            # Normalize embeddings
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(query_embedding, doc_embeddings.T)[0]
            return similarities
            
        except Exception as e:
            logging.warning(f"Semantic similarity computation failed: {e}")
            return np.zeros(len(documents))
            
    def compute_keyword_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute TF-IDF based keyword similarity"""
        if not documents:
            return np.zeros(len(documents))
            
        try:
            # Fit TF-IDF on documents + query
            all_texts = documents + [query]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Get query vector (last one)
            query_vector = tfidf_matrix[-1]
            doc_vectors = tfidf_matrix[:-1]
            
            # Compute cosine similarity
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            return similarities
            
        except Exception as e:
            logging.warning(f"Keyword similarity computation failed: {e}")
            return np.zeros(len(documents))

class ContextAwareRetriever:
    """Context-aware retrieval with conversation history"""
    
    def __init__(self):
        self.conversation_memory = []
        self.session_topics = set()
        self.user_preferences = {}
        
    def update_context(self, query: str, response: str, metadata: Dict = None):
        """Update conversation context"""
        self.conversation_memory.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        })
        
        # Limit memory size
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
            
        # Extract topics from conversation
        self._extract_session_topics(query)
        
    def _extract_session_topics(self, text: str):
        """Extract topics from conversation"""
        # Simple keyword extraction for topics
        mental_health_topics = [
            "anxiety", "depression", "stress", "trauma", "panic", "phobia",
            "sleep", "mood", "anger", "grief", "relationships", "work"
        ]
        
        text_lower = text.lower()
        for topic in mental_health_topics:
            if topic in text_lower:
                self.session_topics.add(topic)
                
    def get_context_relevance(self, query: str, document: str) -> float:
        """Calculate context relevance based on conversation history"""
        if not self.conversation_memory:
            return 0.0
            
        # Check topic continuity
        topic_relevance = 0.0
        for topic in self.session_topics:
            if topic in document.lower():
                topic_relevance += 0.3
                
        # Check recent conversation relevance
        recent_queries = [entry["query"] for entry in self.conversation_memory[-3:]]
        recent_context = " ".join(recent_queries).lower()
        
        doc_lower = document.lower()
        context_overlap = len(set(recent_context.split()) & set(doc_lower.split()))
        context_relevance = min(context_overlap / max(len(recent_context.split()), 1), 1.0)
        
        return min(topic_relevance + context_relevance * 0.5, 1.0)

class EnhancedRAGRetriever:
    """Main enhanced RAG retrieval system"""
    
    def __init__(self, config: Dict = None):
        # Debug: log the incoming config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EnhancedRAGRetriever received config: {config} (type: {type(config)})")
        
        # Merge config with defaults
        default_config = self._default_config()
        if config:
            if isinstance(config, dict):
                default_config.update(config)
            else:
                self.logger.error(f"Config is not a dict: {type(config)}")
        self.config = default_config
        
        self.terminology = MentalHealthTerminology()
        self.intent_classifier = IntentClassifier()
        
        # Use different models for different knowledge bases
        # CBT uses all-mpnet-base-v2 (768d), ICD-11 uses all-MiniLM-L6-v2 (384d)
        self.cbt_semantic_engine = SemanticSimilarityEngine("all-mpnet-base-v2")
        self.icd11_semantic_engine = SemanticSimilarityEngine("all-MiniLM-L6-v2")
        self.context_retriever = ContextAwareRetriever()
        
        # Knowledge base references
        self.cbt_index = None
        self.cbt_metadata = None
        self.icd11_index = None
        self.icd11_metadata = None
        
        # Debug: log the final config
        self.logger.info(f"EnhancedRAGRetriever initialized with config: {self.config}")
        
    def _default_config(self) -> Dict:
        """Default configuration for the retriever"""
        return {
            "semantic_weight": 0.5,
            "keyword_weight": 0.3,
            "context_weight": 0.2,
            "intent_boost": 0.15,
            "urgency_boost": 0.1,
            "min_relevance_threshold": 0.3,
            "max_results": 5,
            "enable_query_expansion": True,
            "enable_intent_filtering": True,
            "enable_context_awareness": True
        }
        
    def load_knowledge_bases(self, cbt_path: str = None, icd11_path: str = None):
        """Load CBT and ICD-11 knowledge bases"""
        # Set default paths if not provided
        if cbt_path is None:
            cbt_path = "CBT_System/cbt_data/embeddings/cbt_index_standard_20250727_143814.faiss"
        if icd11_path is None:
            icd11_path = "embeddings/index.faiss"
            
        # Load CBT knowledge base
        if Path(cbt_path).exists():
            try:
                self.cbt_index = faiss.read_index(cbt_path)
                
                # Load corresponding metadata
                # Handle different naming patterns for CBT metadata
                if 'standard' in cbt_path:
                    metadata_path = cbt_path.replace('cbt_index_', 'cbt_metadata_').replace('.faiss', '.pkl')
                else:
                    metadata_path = cbt_path.replace('.faiss', '.pkl').replace('index', 'metadata')
                
                self.logger.info(f"CBT metadata path: {metadata_path}")
                self.logger.info(f"CBT metadata path exists: {Path(metadata_path).exists()}")
                
                if Path(metadata_path).exists():
                    try:
                        with open(metadata_path, 'rb') as f:
                            self.cbt_metadata = pickle.load(f)
                        self.logger.info(f"CBT metadata loaded successfully, length: {len(self.cbt_metadata) if self.cbt_metadata else 'None'}")
                    except Exception as e:
                        self.logger.error(f"Failed to load CBT metadata: {e}")
                        self.cbt_metadata = None
                else:
                    self.logger.warning(f"CBT metadata file not found: {metadata_path}")
                    self.cbt_metadata = None
                        
                self.logger.info(f"Loaded CBT knowledge base: {self.cbt_index.ntotal} items")
            except Exception as e:
                self.logger.error(f"Failed to load CBT knowledge base: {e}")
                
        # Load ICD-11 knowledge base (if available)
        if Path(icd11_path).exists():
            try:
                self.icd11_index = faiss.read_index(icd11_path)
                
                # Load corresponding metadata
                metadata_path = icd11_path.replace('.faiss', '.pkl')
                if Path(metadata_path).exists():
                    with open(metadata_path, 'rb') as f:
                        metadata_tuple = pickle.load(f)
                        
                    # Handle different metadata formats
                    if isinstance(metadata_tuple, tuple) and len(metadata_tuple) == 2:
                        # This is likely (docstore, index_to_docstore_id) format
                        docstore, index_to_docstore_id = metadata_tuple
                        # Try different ways to access docstore content
                        if hasattr(docstore, '_dict'):
                            # Convert docstore to list format
                            self.icd11_metadata = []
                            for i in range(len(index_to_docstore_id)):
                                doc_id = index_to_docstore_id[i]
                                if doc_id in docstore._dict:
                                    doc = docstore._dict[doc_id]
                                    if hasattr(doc, 'page_content'):
                                        self.icd11_metadata.append({
                                            "content": doc.page_content,
                                            "metadata": getattr(doc, 'metadata', {})
                                        })
                                    else:
                                        self.icd11_metadata.append({
                                            "content": str(doc),
                                            "metadata": {}
                                        })
                                else:
                                    # Fallback for missing documents
                                    self.icd11_metadata.append({
                                        "content": f"Document {doc_id} not found",
                                        "metadata": {}
                                    })
                        elif hasattr(docstore, 'dict'):
                            # Fallback to 'dict' attribute
                            self.icd11_metadata = []
                            for i in range(len(index_to_docstore_id)):
                                doc_id = index_to_docstore_id[i]
                                if doc_id in docstore.dict:
                                    doc = docstore.dict[doc_id]
                                    if hasattr(doc, 'page_content'):
                                        self.icd11_metadata.append({
                                            "content": doc.page_content,
                                            "metadata": getattr(doc, 'metadata', {})
                                        })
                                    else:
                                        self.icd11_metadata.append({
                                            "content": str(doc),
                                            "metadata": {}
                                        })
                                else:
                                    # Fallback for missing documents
                                    self.icd11_metadata.append({
                                        "content": f"Document {doc_id} not found",
                                        "metadata": {}
                                    })
                        else:
                            self.icd11_metadata = metadata_tuple
                    else:
                        self.icd11_metadata = metadata_tuple
                        
                self.logger.info(f"Loaded ICD-11 knowledge base: {self.icd11_index.ntotal} items")
            except Exception as e:
                self.logger.error(f"Failed to load ICD-11 knowledge base: {e}")
                
    def retrieve(self, 
                query: str, 
                context: RetrievalContext = None,
                strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_SEMANTIC_KEYWORD,
                top_k: int = None) -> List[RetrievalResult]:
        """Main retrieval method with enhanced features"""
        
        # Debug: check config type and content
        self.logger.info(f"Config type: {type(self.config)}, Config content: {self.config}")
        
        top_k = top_k or self.config["max_results"]
        context = context or RetrievalContext()
        
        # Step 1: Intent classification
        intent, intent_confidence = self.intent_classifier.classify_intent(query)
        
        # Step 2: Urgency detection
        urgency_level = self.terminology.detect_urgency_level(query)
        context.urgency_level = max(urgency_level, context.urgency_level)
        
        # Step 3: Query expansion
        if self.config["enable_query_expansion"]:
            expanded_terms = self.terminology.expand_query_terms(query)
            expanded_query = " ".join(expanded_terms[:10])  # Limit expansion
        else:
            expanded_query = query
            
        # Step 4: Retrieve from knowledge bases
        all_results = []
        
        # Retrieve from CBT knowledge base
        if self.cbt_index and self.cbt_metadata:
            cbt_results = self._retrieve_from_knowledge_base(
                expanded_query, 
                self.cbt_index, 
                self.cbt_metadata,
                "cbt",
                intent,
                context,
                strategy
            )
            all_results.extend(cbt_results)
            
        # Retrieve from ICD-11 knowledge base
        if self.icd11_index and self.icd11_metadata:
            icd11_results = self._retrieve_from_knowledge_base(
                expanded_query,
                self.icd11_index,
                self.icd11_metadata, 
                "icd11",
                intent,
                context,
                strategy
            )
            all_results.extend(icd11_results)
            
        # Step 5: Fusion and ranking
        final_results = self._fuse_and_rank_results(
            all_results, query, intent, intent_confidence, context
        )
        
        # Step 6: Update context
        if self.config["enable_context_awareness"]:
            top_result = final_results[0] if final_results else None
            self.context_retriever.update_context(
                query, 
                top_result.content if top_result else "",
                {"intent": intent.value, "urgency": urgency_level}
            )
            
        return final_results[:top_k]
        
    def _retrieve_from_knowledge_base(self,
                                    query: str,
                                    index: faiss.Index,
                                    metadata: List[Dict],
                                    source: str,
                                    intent: QueryIntent,
                                    context: RetrievalContext,
                                    strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """Retrieve from a specific knowledge base"""
        
        if not metadata or not index:
            return []
            
        try:
            # Get query embedding using appropriate semantic engine
            if source == "cbt":
                semantic_engine = self.cbt_semantic_engine
            else:  # icd11
                semantic_engine = self.icd11_semantic_engine
                
            query_embedding = semantic_engine.model.encode([query])[0].reshape(1, -1)
            
            # Search in FAISS index
            k = min(self.config["max_results"] * 2, index.ntotal)  # Get more results for filtering
            distances, indices = index.search(query_embedding, k)
            
            # Get metadata for retrieved documents
            retrieved_metadata = []
            for idx in indices[0]:
                if idx < len(metadata):
                    retrieved_metadata.append(metadata[idx])
            
            if not retrieved_metadata:
                return []
            
            # Get documents for similarity computation
            documents = []
            for item in retrieved_metadata:
                if isinstance(item, dict):
                    # Try different content keys
                    content = item.get("text_used_for_embedding", item.get("content", item.get("text", "")))
                    # Clean HTML tags if present
                    if content and isinstance(content, str):
                        import re
                        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
                        content = re.sub(r'\s+', ' ', content)     # Normalize whitespace
                        content = content.strip()
                else:
                    content = str(item)
                documents.append(content)
            
            # Compute different similarity scores using appropriate semantic engine
            semantic_scores = semantic_engine.compute_semantic_similarity(query, documents)
            keyword_scores = semantic_engine.compute_keyword_similarity(query, documents)
            
            results = []
            
            for i, (semantic_score, keyword_score) in enumerate(zip(semantic_scores, keyword_scores)):
                if i >= len(retrieved_metadata):
                    continue
                    
                meta = retrieved_metadata[i]
                
                # Ensure meta is a dictionary
                if not isinstance(meta, dict):
                    meta = {"content": str(meta)}
                
                # Compute context relevance
                context_score = 0.0
                if self.config["enable_context_awareness"]:
                    context_score = self.context_retriever.get_context_relevance(
                        query, meta.get("content", "")
                    )
                    
                # Intent matching score
                intent_score = self._compute_intent_match(meta, intent)
                
                # Combined relevance score
                relevance_score = (
                    semantic_score * self.config["semantic_weight"] +
                    keyword_score * self.config["keyword_weight"] +
                    context_score * self.config["context_weight"] +
                    intent_score * self.config["intent_boost"]
                )
                
                # Apply urgency boost
                if context.urgency_level > 3:
                    relevance_score += self.config["urgency_boost"]
                    
                # Filter by minimum threshold
                if relevance_score >= self.config["min_relevance_threshold"]:
                    result = RetrievalResult(
                        content=meta.get("content", ""),
                        source=source,
                        relevance_score=relevance_score,
                        confidence=semantic_score,
                        intent_match=intent_score,
                        context_relevance=context_score,
                        metadata=meta,
                        reasoning=f"Semantic: {semantic_score:.3f}, Keyword: {keyword_score:.3f}, Context: {context_score:.3f}, Intent: {intent_score:.3f}"
                    )
                    results.append(result)
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error in _retrieve_from_knowledge_base: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        
    def _compute_intent_match(self, metadata: Dict, intent: QueryIntent) -> float:
        """Compute how well a document matches the query intent"""
        
        # Get document category/type information
        classification = metadata.get("classification", {})
        primary_category = classification.get("primary_category", "")
        content_type = classification.get("content_type", "")
        
        # Intent-category matching rules
        intent_matches = {
            QueryIntent.SKILL_LEARNING: {
                "categories": ["cognitive_restructuring", "relaxation_techniques", "behavioral_activation"],
                "content_types": ["step_by_step_guide", "worksheet", "technique_description"]
            },
            QueryIntent.EMOTIONAL_SUPPORT: {
                "categories": ["psychoeducation", "relaxation_techniques"],
                "content_types": ["case_study", "technique_description"]
            },
            QueryIntent.SYMPTOM_ASSESSMENT: {
                "categories": ["psychoeducation"],
                "content_types": ["assessment_tool", "technique_description"]
            },
            QueryIntent.TREATMENT_GUIDANCE: {
                "categories": ["exposure_therapy", "problem_solving", "cognitive_restructuring"],
                "content_types": ["step_by_step_guide", "technique_description"]
            },
            QueryIntent.INFORMATION_SEEKING: {
                "categories": ["psychoeducation"],
                "content_types": ["technique_description", "case_study"]
            }
        }
        
        if intent in intent_matches:
            matches = intent_matches[intent]
            category_match = 1.0 if primary_category in matches["categories"] else 0.0
            type_match = 1.0 if content_type in matches["content_types"] else 0.0
            return (category_match + type_match) / 2
            
        return 0.0
        
    def _fuse_and_rank_results(self,
                              results: List[RetrievalResult],
                              query: str,
                              intent: QueryIntent,
                              intent_confidence: float,
                              context: RetrievalContext) -> List[RetrievalResult]:
        """Fuse and rank results from multiple sources"""
        
        if not results:
            return []
            
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity filtering (avoid too many similar results)
        filtered_results = self._apply_diversity_filtering(results)
        
        # Apply intent-based reranking
        if intent_confidence > 0.5:
            filtered_results = self._apply_intent_reranking(filtered_results, intent)
            
        return filtered_results
        
    def _apply_diversity_filtering(self, results: List[RetrievalResult], similarity_threshold: float = 0.85) -> List[RetrievalResult]:
        """Remove overly similar results to increase diversity"""
        
        if len(results) <= 1:
            return results
            
        filtered = [results[0]]  # Always keep the top result
        
        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            for selected in filtered:
                if self._compute_result_similarity(result, selected) > similarity_threshold:
                    is_diverse = False
                    break
                    
            if is_diverse:
                filtered.append(result)
                
        return filtered
        
    def _compute_result_similarity(self, result1: RetrievalResult, result2: RetrievalResult) -> float:
        """Compute similarity between two results"""
        # Simple word overlap similarity
        words1 = set(result1.content.lower().split())
        words2 = set(result2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
        
    def _apply_intent_reranking(self, results: List[RetrievalResult], intent: QueryIntent) -> List[RetrievalResult]:
        """Rerank results based on intent matching"""
        
        # Boost results that match the intent well
        for result in results:
            if result.intent_match > 0.5:
                result.relevance_score *= 1.2
                
        # Re-sort after boosting
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results

if __name__ == "__main__":
    # Example usage
    retriever = EnhancedRAGRetriever()
    
    # Example query
    query = "I'm feeling anxious and need some coping strategies"
    context = RetrievalContext(urgency_level=2)
    
    results = retriever.retrieve(query, context)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result.source}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        print(f"   Content: {result.content[:200]}...")
        print(f"   Reasoning: {result.reasoning}") 