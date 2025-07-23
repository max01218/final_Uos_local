#!/usr/bin/env python3
"""
CBT Data Processor
Cleans and standardizes collected CBT data
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

class CBTDataProcessor:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.setup_logging()
        
        # CBT technique categories
        self.technique_categories = {
            "cognitive_restructuring": [
                "thought challenging", "cognitive distortion", "automatic thoughts",
                "thought record", "thinking errors", "negative thinking",
                "cognitive bias", "balanced thinking"
            ],
            "behavioral_activation": [
                "activity scheduling", "behavioral experiment", "pleasant activities",
                "activity monitoring", "behavioral change", "action plan",
                "goal setting", "activity diary"
            ],
            "exposure_therapy": [
                "exposure", "systematic desensitization", "hierarchy",
                "fear ladder", "gradual exposure", "flooding",
                "imaginal exposure", "in vivo exposure"
            ],
            "problem_solving": [
                "problem solving", "solution focused", "decision making",
                "coping strategies", "stress management", "conflict resolution"
            ],
            "relaxation_techniques": [
                "relaxation", "deep breathing", "progressive muscle",
                "mindfulness", "meditation", "breathing exercises"
            ],
            "psychoeducation": [
                "education", "understanding", "learning about",
                "information", "awareness", "knowledge"
            ]
        }
        
        # Assessment tools patterns
        self.assessment_patterns = {
            "mood_scales": [
                "depression scale", "anxiety scale", "mood questionnaire",
                "phq", "gad", "beck inventory", "hamilton rating"
            ],
            "thought_monitoring": [
                "thought diary", "thought record", "cognitive monitoring",
                "thinking patterns", "automatic thoughts log"
            ],
            "behavioral_tracking": [
                "activity log", "behavior diary", "mood tracking",
                "sleep diary", "exercise log"
            ]
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_raw_data(self) -> List[Dict]:
        """Load all raw collected data"""
        raw_data = []
        raw_dir = self.base_dir / "raw_data" / "government"
        
        for file_path in raw_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        raw_data.extend(data)
                    else:
                        raw_data.append(data)
                        
                self.logger.info(f"Loaded data from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                
        return raw_data
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
            
        # Remove extra whitespace but preserve single spaces between words
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and ensure spaces around words
        text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\(\)]', ' ', text)
        
        # Fix cases where punctuation is directly attached to words without space
        text = re.sub(r'([a-zA-Z])([.!?:;,])', r'\1 \2', text)
        text = re.sub(r'([.!?:;,])([a-zA-Z])', r'\1 \2', text)
        
        # Remove repetitive patterns but be more careful
        text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)
        
        # Clean up common web artifacts
        web_artifacts = [
            r'cookie policy',
            r'privacy policy', 
            r'terms of service',
            r'skip to main content',
            r'search this website',
            r'share this page',
            r'print this page',
            r'last updated',
            r'page last reviewed'
        ]
        
        for artifact in web_artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)
        
        # Final cleanup: remove multiple spaces and trim
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def extract_techniques(self, content: str) -> List[Dict]:
        """Extract CBT techniques from content"""
        techniques = []
        content_lower = content.lower()
        
        for category, keywords in self.technique_categories.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Find context around the technique
                    context = self.extract_context(content, keyword)
                    
                    if context and len(context) > 50:  # Meaningful context
                        techniques.append({
                            "category": category,
                            "technique": keyword,
                            "context": context,
                            "confidence": self.calculate_confidence(content, keyword)
                        })
                        
        return techniques
        
    def extract_context(self, content: str, keyword: str, window=300) -> str:
        """Extract context around a keyword"""
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        pos = content_lower.find(keyword_lower)
        if pos == -1:
            return ""
            
        start = max(0, pos - window)
        end = min(len(content), pos + len(keyword) + window)
        
        context = content[start:end]
        
        # Clean context with improved text processing
        context = self.clean_text(context)
        
        # Additional cleaning for context extraction
        # Fix common OCR/extraction errors
        context = re.sub(r'([a-z])([A-Z])', r'\1 \2', context)  # Fix camelCase
        context = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', context)  # Fix letter-number joins
        context = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', context)  # Fix number-letter joins
        
        # Fix common word concatenations
        context = re.sub(r'([a-z])(that|with|and|the|for|to|of|in|on|at)', r'\1 \2', context)
        context = re.sub(r'(that|with|and|the|for|to|of|in|on|at)([A-Z][a-z])', r'\1 \2', context)
        
        # Clean up multiple spaces again
        context = re.sub(r'\s+', ' ', context)
        
        # Try to start and end at sentence boundaries
        sentences = re.split(r'[.!?]', context)
        if len(sentences) > 2:
            # Take the middle sentences to avoid partial sentences at start/end
            middle_sentences = [s.strip() for s in sentences[1:-1] if s.strip()]
            if middle_sentences:
                context = '. '.join(middle_sentences) + '.'
            
        return context.strip()
        
    def calculate_confidence(self, content: str, keyword: str) -> float:
        """Calculate confidence score for technique extraction"""
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        # Count occurrences
        occurrences = content_lower.count(keyword_lower)
        
        # Check for surrounding context
        context_indicators = [
            'step', 'method', 'technique', 'approach', 'strategy',
            'exercise', 'practice', 'homework', 'assignment'
        ]
        
        context_score = sum(
            1 for indicator in context_indicators 
            if indicator in content_lower
        )
        
        # Calculate confidence (0.0 to 1.0)
        confidence = min(1.0, (occurrences * 0.2) + (context_score * 0.1))
        
        return round(confidence, 2)
        
    def extract_assessments(self, content: str) -> List[Dict]:
        """Extract assessment tools from content"""
        assessments = []
        content_lower = content.lower()
        
        for category, keywords in self.assessment_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    context = self.extract_context(content, keyword)
                    
                    if context and len(context) > 30:
                        assessments.append({
                            "category": category,
                            "assessment": keyword,
                            "context": context,
                            "confidence": self.calculate_confidence(content, keyword)
                        })
                        
        return assessments
        
    def categorize_content(self, item: Dict) -> Dict:
        """Categorize content into CBT components"""
        content = item.get('content', '')
        title = item.get('title', '')
        
        # Extract techniques and assessments
        techniques = self.extract_techniques(content)
        assessments = self.extract_assessments(content)
        
        # Determine primary category
        primary_category = "general"
        max_techniques = 0
        
        category_counts = {}
        for technique in techniques:
            cat = technique['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        if category_counts:
            primary_category = max(category_counts, key=category_counts.get)
            max_techniques = category_counts[primary_category]
            
        return {
            "url": item.get('url', ''),
            "source": item.get('source', ''),
            "title": self.clean_text(title),
            "content": self.clean_text(content),
            "primary_category": primary_category,
            "techniques": techniques,
            "assessments": assessments,
            "word_count": len(content.split()),
            "quality_score": self.calculate_quality_score(item, techniques, assessments),
            "collected_at": item.get('collected_at', 0),
            "processed_at": datetime.now().isoformat()
        }
        
    def calculate_quality_score(self, item: Dict, techniques: List, assessments: List) -> float:
        """Calculate quality score for content"""
        score = 0.0
        
        # Content length score (0.0 - 0.3)
        word_count = len(item.get('content', '').split())
        if word_count > 100:
            score += min(0.3, word_count / 1000)
            
        # Technique diversity score (0.0 - 0.4)
        unique_categories = len(set(t['category'] for t in techniques))
        score += min(0.4, unique_categories * 0.1)
        
        # Assessment tools score (0.0 - 0.2)
        if assessments:
            score += min(0.2, len(assessments) * 0.05)
            
        # Source credibility score (0.0 - 0.1)
        source = item.get('source', '')
        if 'nhs' in source or 'nimh' in source or 'cci' in source:
            score += 0.1
            
        return round(min(1.0, score), 2)
        
    def deduplicate_content(self, processed_data: List[Dict]) -> List[Dict]:
        """Remove duplicate and very similar content"""
        unique_content = []
        seen_urls = set()
        
        for item in processed_data:
            url = item.get('url', '')
            
            # Skip exact URL duplicates
            if url in seen_urls:
                continue
                
            # Check content similarity
            is_duplicate = False
            content = item.get('content', '')
            
            for existing_item in unique_content:
                existing_content = existing_item.get('content', '')
                similarity = self.calculate_similarity(content, existing_content)
                
                if similarity > 0.85:  # 85% similar
                    is_duplicate = True
                    # Keep the one with higher quality score
                    if item['quality_score'] > existing_item['quality_score']:
                        unique_content.remove(existing_item)
                        unique_content.append(item)
                    break
                    
            if not is_duplicate:
                unique_content.append(item)
                seen_urls.add(url)
                
        self.logger.info(f"Deduplicated {len(processed_data)} -> {len(unique_content)} items")
        return unique_content
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def save_processed_data(self, processed_data: List[Dict]):
        """Save processed data to structured format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all processed data
        all_data_path = self.base_dir / "raw_data" / "processed" / f"cbt_processed_{timestamp}.json"
        
        with open(all_data_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
        # Save by category
        categories = {}
        for item in processed_data:
            category = item['primary_category']
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
            
        for category, items in categories.items():
            category_path = self.base_dir / "structured_data" / "techniques" / f"{category}_{timestamp}.json"
            
            with open(category_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
                
        self.logger.info(f"Saved processed data: {len(processed_data)} total items")
        self.logger.info(f"Categories: {list(categories.keys())}")
        
        return all_data_path
        
    def process_all_data(self) -> str:
        """Main processing pipeline"""
        self.logger.info("Starting CBT data processing")
        
        # Load raw data
        raw_data = self.load_raw_data()
        self.logger.info(f"Loaded {len(raw_data)} raw items")
        
        if not raw_data:
            self.logger.warning("No raw data found to process")
            return ""
            
        # Process each item
        processed_data = []
        for item in raw_data:
            try:
                processed_item = self.categorize_content(item)
                if processed_item['quality_score'] > 0.1:  # Filter low quality
                    processed_data.append(processed_item)
                    
            except Exception as e:
                self.logger.error(f"Failed to process item: {e}")
                
        # Deduplicate
        unique_data = self.deduplicate_content(processed_data)
        
        # Save processed data
        output_path = self.save_processed_data(unique_data)
        
        self.logger.info(f"Processing completed: {len(unique_data)} unique items")
        return str(output_path)

def main():
    """Main processing function"""
    print("CBT Data Processing System")
    print("=" * 40)
    
    processor = CBTDataProcessor()
    
    try:
        output_path = processor.process_all_data()
        
        if output_path:
            print(f"\nProcessing completed successfully!")
            print(f"Output saved to: {output_path}")
        else:
            print("No data to process")
            
    except Exception as e:
        print(f"Processing failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    main() 