#!/usr/bin/env python3
"""
CBT Integration Module
Integrates CBT functionality with existing systems
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Try to import required libraries
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependencies: {e}")
    faiss = None
    SentenceTransformer = None

class CBTKnowledgeBase:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.setup_logging()
        
        # Load CBT knowledge base
        self.load_cbt_knowledge()
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("cbt_integration")
        
    def load_cbt_knowledge(self) -> bool:
        """Load CBT vector index and metadata"""
        try:
            embeddings_dir = self.base_dir / "embeddings"
            
            # Check if CBT index exists
            index_path = embeddings_dir / "cbt_index.faiss"
            metadata_path = embeddings_dir / "cbt_metadata.pkl"
            summary_path = embeddings_dir / "cbt_index_summary.json"
            
            if not (index_path.exists() and metadata_path.exists()):
                self.logger.warning("CBT knowledge base not found")
                return False
                
            # Load summary info
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                
            model_name = summary.get('embedding_model', 'all-MiniLM-L6-v2')
            
            # Load embedding model
            if SentenceTransformer:
                self.embedding_model = SentenceTransformer(model_name)
                self.logger.info(f"Loaded CBT embedding model: {model_name}")
            else:
                self.logger.error("SentenceTransformer not available")
                return False
                
            # Load FAISS index
            if faiss:
                self.index = faiss.read_index(str(index_path))
                self.logger.info(f"Loaded CBT FAISS index with {self.index.ntotal} vectors")
            else:
                self.logger.error("FAISS not available")
                return False
                
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                
            self.logger.info(f"Loaded CBT metadata: {len(self.metadata)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CBT knowledge base: {e}")
            return False
            
    def search_cbt_techniques(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """Search for CBT techniques and information"""
        if not self.index or not self.embedding_model:
            self.logger.warning("CBT knowledge base not available")
            return []
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in index
            search_k = top_k * 3  # Get more results for filtering
            scores, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['relevance_score'] = float(score)
                    
                    # Apply category filter if specified
                    if category_filter and result.get('category') != category_filter:
                        continue
                        
                    results.append(result)
                    
                    if len(results) >= top_k:
                        break
                        
            return results
            
        except Exception as e:
            self.logger.error(f"CBT search failed: {e}")
            return []
            
    def get_cbt_recommendation(self, user_query: str, context: str = "") -> Dict:
        """Get CBT technique recommendations based on user query"""
        
        # Analyze query for key indicators
        query_analysis = self.analyze_user_query(user_query)
        
        # Search for relevant CBT techniques
        search_query = f"{user_query} {' '.join(query_analysis['indicators'])}"
        search_results = self.search_cbt_techniques(search_query, top_k=3)
        
        # Filter by technique type
        technique_results = [r for r in search_results if r.get('type') == 'cbt_technique']
        content_results = [r for r in search_results if r.get('type') == 'main_content']
        
        return {
            'query_analysis': query_analysis,
            'recommended_techniques': technique_results[:2],
            'supporting_content': content_results[:2],
            'total_found': len(search_results)
        }
        
    def analyze_user_query(self, query: str) -> Dict:
        """Analyze user query to identify relevant CBT indicators"""
        query_lower = query.lower()
        
        # CBT condition indicators
        condition_indicators = {
            'depression': ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'down'],
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'stress'],
            'trauma': ['trauma', 'ptsd', 'flashback', 'nightmare', 'abuse'],
            'ocd': ['obsessive', 'compulsive', 'ritual', 'checking', 'washing'],
            'phobia': ['phobia', 'afraid', 'scared', 'avoidance'],
            'anger': ['angry', 'rage', 'furious', 'irritated', 'mad']
        }
        
        # CBT technique indicators
        technique_indicators = {
            'cognitive_restructuring': ['thoughts', 'thinking', 'believe', 'assume'],
            'behavioral_activation': ['activity', 'behavior', 'action', 'doing'],
            'exposure': ['avoid', 'escape', 'fear', 'exposure'],
            'relaxation': ['tense', 'relaxation', 'calm', 'breathe'],
            'problem_solving': ['problem', 'solution', 'decision', 'choice']
        }
        
        detected_conditions = []
        detected_techniques = []
        confidence_scores = {}
        
        # Check for condition indicators
        for condition, keywords in condition_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                detected_conditions.append(condition)
                confidence_scores[condition] = score / len(keywords)
                
        # Check for technique indicators  
        for technique, keywords in technique_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                detected_techniques.append(technique)
                confidence_scores[technique] = score / len(keywords)
                
        return {
            'detected_conditions': detected_conditions,
            'suggested_techniques': detected_techniques,
            'confidence_scores': confidence_scores,
            'indicators': detected_conditions + detected_techniques
        }
        
    def format_cbt_response(self, recommendations: Dict, user_query: str) -> str:
        """Format CBT recommendations into a response"""
        
        if not recommendations['recommended_techniques']:
            return "I understand you're looking for support. While I can provide general guidance, I recommend speaking with a mental health professional for personalized CBT techniques."
            
        response_parts = []
        
        # Acknowledge the user's situation
        conditions = recommendations['query_analysis']['detected_conditions']
        if conditions:
            condition_text = ', '.join(conditions)
            response_parts.append(f"I understand you're dealing with {condition_text}.")
            
        # Present CBT techniques
        techniques = recommendations['recommended_techniques']
        if techniques:
            response_parts.append("Here are some evidence-based CBT techniques that may help:")
            
            for i, technique in enumerate(techniques, 1):
                technique_name = technique['metadata'].get('technique_name', 'CBT technique')
                category = technique['metadata'].get('technique_category', '')
                
                # Extract key points from the technique context
                context = technique.get('text', '')
                key_points = self.extract_key_points(context)
                
                response_parts.append(f"\n{i}. {technique_name.title()}")
                if key_points:
                    response_parts.append(f"   {key_points}")
                    
        # Add supporting information
        content = recommendations['supporting_content']
        if content and len(content) > 0:
            best_content = content[0]
            additional_info = self.extract_key_points(best_content.get('text', ''))
            if additional_info:
                response_parts.append(f"\nAdditional guidance: {additional_info}")
                
        # Add professional disclaimer
        response_parts.append("\nRemember: These are general techniques. For personalized treatment, please consult a qualified mental health professional.")
        
        return ' '.join(response_parts)
        
    def extract_key_points(self, text: str, max_length: int = 150) -> str:
        """Extract key points from CBT technique text"""
        if not text:
            return ""
            
        # Remove "CBT Technique:" prefix if present
        text = text.replace('CBT Technique:', '').strip()
        
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Find most informative sentences (containing action words)
        action_words = ['practice', 'try', 'identify', 'challenge', 'focus', 'write', 'schedule', 'engage']
        
        key_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in action_words):
                key_sentences.append(sentence)
                
        # If no action sentences, take first meaningful sentences
        if not key_sentences:
            key_sentences = [s for s in sentences if len(s) > 20][:2]
            
        # Combine and truncate
        result = '. '.join(key_sentences[:2])
        if len(result) > max_length:
            result = result[:max_length].rsplit(' ', 1)[0] + '...'
            
        return result
        
    def is_available(self) -> bool:
        """Check if CBT knowledge base is available"""
        return self.index is not None and self.embedding_model is not None

class CBTIntegration:
    def __init__(self, base_dir="cbt_data"):
        self.cbt_kb = CBTKnowledgeBase(base_dir)
        self.logger = logging.getLogger("cbt_integration")
        
    def enhance_response_with_cbt(self, user_query: str, context: str, base_response: str) -> str:
        """Enhance a response with CBT techniques if relevant"""
        
        if not self.cbt_kb.is_available():
            return base_response
            
        # Check if query would benefit from CBT techniques
        if self.should_include_cbt(user_query):
            try:
                recommendations = self.cbt_kb.get_cbt_recommendation(user_query, context)
                
                if recommendations['recommended_techniques']:
                    cbt_response = self.cbt_kb.format_cbt_response(recommendations, user_query)
                    
                    # Combine base response with CBT recommendations
                    enhanced_response = f"{base_response}\n\n{cbt_response}"
                    return enhanced_response
                    
            except Exception as e:
                self.logger.error(f"CBT enhancement failed: {e}")
                
        return base_response
        
    def should_include_cbt(self, query: str) -> bool:
        """Determine if query would benefit from CBT techniques"""
        query_lower = query.lower()
        
        # Keywords that suggest CBT might be helpful
        cbt_indicators = [
            'how can i', 'what should i do', 'help me', 'strategies',
            'cope', 'coping', 'deal with', 'manage', 'overcome',
            'techniques', 'methods', 'ways to', 'exercises'
        ]
        
        # Mental health conditions that commonly use CBT
        condition_keywords = [
            'anxiety', 'depression', 'stress', 'worry', 'panic',
            'thoughts', 'thinking', 'behavior', 'mood', 'fear'
        ]
        
        has_cbt_indicator = any(indicator in query_lower for indicator in cbt_indicators)
        has_condition_keyword = any(keyword in query_lower for keyword in condition_keywords)
        
        return has_cbt_indicator and has_condition_keyword
        
    def get_cbt_status(self) -> Dict:
        """Get CBT integration status"""
        return {
            'available': self.cbt_kb.is_available(),
            'total_techniques': len([m for m in self.cbt_kb.metadata if m.get('type') == 'cbt_technique']),
            'total_content': len([m for m in self.cbt_kb.metadata if m.get('type') == 'main_content']),
            'categories': list(set(m.get('category', 'unknown') for m in self.cbt_kb.metadata))
        }

def main():
    """Test CBT integration"""
    print("CBT Integration Test")
    print("=" * 40)
    
    integration = CBTIntegration()
    status = integration.get_cbt_status()
    
    print(f"CBT Available: {status['available']}")
    if status['available']:
        print(f"CBT Techniques: {status['total_techniques']}")
        print(f"Content Items: {status['total_content']}")
        print(f"Categories: {status['categories']}")
        
        # Test queries
        test_queries = [
            "I feel anxious all the time, what can I do?",
            "How can I stop negative thinking?",
            "I'm depressed and need coping strategies"
        ]
        
        for query in test_queries:
            print(f"\nTest Query: {query}")
            
            if integration.should_include_cbt(query):
                recommendations = integration.cbt_kb.get_cbt_recommendation(query)
                response = integration.cbt_kb.format_cbt_response(recommendations, query)
                print(f"CBT Response: {response[:200]}...")
            else:
                print("No CBT enhancement needed")
    else:
        print("CBT knowledge base not available")
        print("Run the data collection and vectorization scripts first")

if __name__ == "__main__":
    main() 