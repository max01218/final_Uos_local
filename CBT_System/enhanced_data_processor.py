#!/usr/bin/env python3
"""
Enhanced CBT Data Processing System
Advanced processing with quality assessment, deduplication, and intelligent classification
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from datetime import datetime
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EnhancedCBTDataProcessor:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.setup_logging()
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced CBT technique taxonomies
        self.technique_taxonomy = {
            "cognitive_restructuring": {
                "keywords": [
                    "thought challenging", "cognitive distortion", "automatic thoughts",
                    "thought record", "thinking errors", "negative thinking",
                    "cognitive bias", "balanced thinking", "cognitive reframing",
                    "dysfunctional thoughts", "catastrophizing", "all or nothing"
                ],
                "subcategories": {
                    "thought_identification": ["automatic thoughts", "thought record", "thought diary"],
                    "distortion_recognition": ["cognitive distortion", "thinking errors", "cognitive bias"],
                    "thought_challenging": ["thought challenging", "balanced thinking", "evidence examination"],
                    "cognitive_reframing": ["cognitive reframing", "perspective taking", "alternative thoughts"]
                }
            },
            "behavioral_activation": {
                "keywords": [
                    "activity scheduling", "behavioral experiment", "pleasant activities",
                    "activity monitoring", "behavioral change", "action plan",
                    "goal setting", "activity diary", "behavioral activation",
                    "activity log", "engagement", "mastery activities"
                ],
                "subcategories": {
                    "activity_scheduling": ["activity scheduling", "activity planning", "daily schedule"],
                    "behavioral_experiments": ["behavioral experiment", "hypothesis testing", "behavioral test"],
                    "pleasant_activities": ["pleasant activities", "enjoyable activities", "reward activities"],
                    "goal_setting": ["goal setting", "action plan", "behavioral goals"]
                }
            },
            "exposure_therapy": {
                "keywords": [
                    "exposure", "systematic desensitization", "hierarchy",
                    "fear ladder", "gradual exposure", "flooding",
                    "imaginal exposure", "in vivo exposure", "exposure exercises"
                ],
                "subcategories": {
                    "gradual_exposure": ["gradual exposure", "systematic desensitization", "step by step"],
                    "exposure_hierarchy": ["hierarchy", "fear ladder", "exposure ladder"],
                    "in_vivo_exposure": ["in vivo exposure", "real life exposure", "actual exposure"],
                    "imaginal_exposure": ["imaginal exposure", "imagery exposure", "mental exposure"]
                }
            },
            "problem_solving": {
                "keywords": [
                    "problem solving", "solution focused", "decision making",
                    "coping strategies", "stress management", "conflict resolution",
                    "problem identification", "brainstorming", "solution implementation"
                ],
                "subcategories": {
                    "problem_identification": ["problem identification", "problem definition", "issue clarification"],
                    "solution_generation": ["brainstorming", "solution generation", "alternative solutions"],
                    "decision_making": ["decision making", "option evaluation", "choice selection"],
                    "implementation": ["solution implementation", "action steps", "follow through"]
                }
            },
            "relaxation_techniques": {
                "keywords": [
                    "relaxation", "deep breathing", "progressive muscle",
                    "mindfulness", "meditation", "breathing exercises",
                    "relaxation training", "muscle relaxation", "guided imagery"
                ],
                "subcategories": {
                    "breathing_techniques": ["deep breathing", "breathing exercises", "diaphragmatic breathing"],
                    "muscle_relaxation": ["progressive muscle", "muscle relaxation", "tension release"],
                    "mindfulness": ["mindfulness", "present moment", "mindful awareness"],
                    "guided_imagery": ["guided imagery", "visualization", "mental imagery"]
                }
            },
            "psychoeducation": {
                "keywords": [
                    "education", "understanding", "learning about",
                    "information", "awareness", "knowledge", "explanation",
                    "psychoeducation", "educational material", "information sharing"
                ],
                "subcategories": {
                    "disorder_education": ["disorder information", "condition education", "symptom explanation"],
                    "treatment_education": ["treatment information", "therapy explanation", "intervention education"],
                    "skill_education": ["skill learning", "technique education", "strategy teaching"],
                    "self_awareness": ["self awareness", "insight development", "understanding patterns"]
                }
            }
        }
        
        # Quality assessment criteria
        self.quality_metrics = {
            "content_length": {"min": 100, "optimal": 500, "max": 3000},
            "sentence_count": {"min": 5, "optimal": 15, "max": 50},
            "readability": {"min_score": 30, "max_score": 80},
            "professional_terms": {"min_count": 2, "optimal_count": 8},
            "structure_indicators": ["step", "stage", "phase", "first", "second", "next", "finally"],
            "technical_depth": ["research", "study", "evidence", "clinical", "systematic", "randomized"]
        }
        
        # Content type patterns
        self.content_type_patterns = {
            "technique_description": [
                r"technique.*?(?:involves|includes|consists)",
                r"method.*?(?:involves|includes|consists)",
                r"approach.*?(?:involves|includes|consists)"
            ],
            "step_by_step_guide": [
                r"step \d+[:.]",
                r"\d+[\.\)] .*?(?:first|then|next|finally)",
                r"(?:first|second|third|fourth|fifth|next|then|finally).*?step"
            ],
            "assessment_tool": [
                r"questionnaire", r"scale", r"inventory", r"assessment",
                r"measure", r"rating", r"checklist"
            ],
            "worksheet": [
                r"worksheet", r"exercise", r"practice", r"homework",
                r"activity.*?(?:sheet|form|template)"
            ],
            "case_study": [
                r"case study", r"example.*?(?:patient|client|individual)",
                r"(?:patient|client).*?example"
            ]
        }
        
    def setup_logging(self):
        """Enhanced logging setup"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # File handler
        file_handler = logging.FileHandler(self.base_dir / 'enhanced_processing.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        
        # Setup logger
        self.logger = logging.getLogger("EnhancedCBTProcessor")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_min_enhanced_score(self) -> float:
        """Load minimum enhanced score from config file"""
        config_file = "enhanced_config.json"
        default_score = 40
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                score = config.get("data_processing", {}).get("quality_assessment", {}).get("minimum_enhanced_score", default_score)
                return score
            else:
                return default_score
                
        except Exception as e:
            self.logger.warning(f"Failed to load config, using default score {default_score}: {e}")
            return default_score
        
    def load_enhanced_raw_data(self) -> List[Dict]:
        """Load all enhanced raw data from multiple source types"""
        all_data = []
        
        # Define source directories
        source_dirs = [
            "raw_data/government",
            "raw_data/professional_orgs",
            "raw_data/international_orgs",
            "raw_data/clinical_guidelines",
            "raw_data/academic",
            "raw_data/nonprofit"
        ]
        
        for source_dir in source_dirs:
            dir_path = self.base_dir / source_dir
            if dir_path.exists():
                self.logger.info(f"Loading data from {source_dir}")
                
                for file_path in dir_path.glob("*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                            
                        # Handle both old and new formats
                        if isinstance(file_data, dict) and "data" in file_data:
                            # New format with metadata
                            data_items = file_data["data"]
                            metadata = file_data.get("metadata", {})
                            
                            for item in data_items:
                                item["file_metadata"] = metadata
                                all_data.append(item)
                                
                        elif isinstance(file_data, list):
                            # Old format - list of items
                            all_data.extend(file_data)
                            
                        elif isinstance(file_data, dict):
                            # Single item
                            all_data.append(file_data)
                            
                        self.logger.info(f"Loaded data from {file_path}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load {file_path}: {e}")
                        
        self.logger.info(f"Total raw data items loaded: {len(all_data)}")
        return all_data
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
            
    def advanced_deduplication(self, data: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
        """Advanced deduplication using content similarity and hash matching"""
        self.logger.info("Starting advanced deduplication process")
        
        # Step 1: Exact hash deduplication
        hash_map = {}
        exact_duplicates_removed = 0
        
        for item in data:
            content = item.get("content", "")
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash not in hash_map:
                hash_map[content_hash] = item
            else:
                exact_duplicates_removed += 1
                
        deduplicated_data = list(hash_map.values())
        self.logger.info(f"Removed {exact_duplicates_removed} exact duplicates")
        
        # Step 2: Semantic similarity deduplication
        if len(deduplicated_data) > 1:
            similarity_matrix = np.zeros((len(deduplicated_data), len(deduplicated_data)))
            
            # Calculate pairwise similarities
            for i in range(len(deduplicated_data)):
                for j in range(i + 1, len(deduplicated_data)):
                    content1 = deduplicated_data[i].get("content", "")
                    content2 = deduplicated_data[j].get("content", "")
                    
                    similarity = self.calculate_text_similarity(content1, content2)
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
                    
            # Remove similar items (keep higher quality ones)
            to_remove = set()
            semantic_duplicates_removed = 0
            
            for i in range(len(deduplicated_data)):
                if i in to_remove:
                    continue
                    
                for j in range(i + 1, len(deduplicated_data)):
                    if j in to_remove:
                        continue
                        
                    if similarity_matrix[i][j] > similarity_threshold:
                        # Keep the higher quality item
                        score1 = deduplicated_data[i].get("quality_assessment", {}).get("score", 0)
                        score2 = deduplicated_data[j].get("quality_assessment", {}).get("score", 0)
                        
                        if score1 >= score2:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            
                        semantic_duplicates_removed += 1
                        
            # Filter out removed items
            final_data = [item for i, item in enumerate(deduplicated_data) if i not in to_remove]
            self.logger.info(f"Removed {semantic_duplicates_removed} semantic duplicates")
            
        else:
            final_data = deduplicated_data
            
        self.logger.info(f"Deduplication complete: {len(data)} -> {len(final_data)} items")
        return final_data
        
    def extract_content_features(self, content: str) -> Dict:
        """Extract detailed features from content for classification"""
        features = {
            "word_count": len(content.split()),
            "sentence_count": len(sent_tokenize(content)),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "avg_sentence_length": 0,
            "readability_score": 0,
            "professional_term_density": 0,
            "structure_indicators": [],
            "key_concepts": [],
            "content_type_scores": {},
            "technique_category_scores": {}
        }
        
        sentences = sent_tokenize(content)
        if sentences:
            features["avg_sentence_length"] = sum(len(s.split()) for s in sentences) / len(sentences)
            
        # Simple readability score (Flesch-like)
        if features["sentence_count"] > 0 and features["word_count"] > 0:
            avg_sentence_length = features["word_count"] / features["sentence_count"]
            features["readability_score"] = 206.835 - (1.015 * avg_sentence_length)
            
        # Professional term density
        content_lower = content.lower()
        professional_terms = [
            "intervention", "therapeutic", "evidence-based", "clinical", "research",
            "systematic", "randomized", "efficacy", "assessment", "diagnosis",
            "treatment", "therapy", "cognitive", "behavioral", "psychological"
        ]
        
        prof_term_count = sum(1 for term in professional_terms if term in content_lower)
        features["professional_term_density"] = prof_term_count / features["word_count"] * 100
        
        # Structure indicators
        for indicator in self.quality_metrics["structure_indicators"]:
            if indicator in content_lower:
                features["structure_indicators"].append(indicator)
                
        # Content type classification
        for content_type, patterns in self.content_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
                
            features["content_type_scores"][content_type] = score
            
        # Technique category scoring
        for category, category_data in self.technique_taxonomy.items():
            score = 0
            matched_keywords = []
            
            for keyword in category_data["keywords"]:
                if keyword in content_lower:
                    score += 1
                    matched_keywords.append(keyword)
                    
            features["technique_category_scores"][category] = {
                "score": score,
                "matched_keywords": matched_keywords
            }
            
        return features
        
    def intelligent_classification(self, item: Dict) -> Dict:
        """Intelligent classification of CBT content"""
        content = item.get("content", "")
        features = self.extract_content_features(content)
        
        classification = {
            "primary_category": None,
            "secondary_categories": [],
            "content_type": None,
            "confidence_score": 0,
            "extracted_features": features,
            "subcategory_classifications": {},
            "quality_indicators": {}
        }
        
        # Determine primary category
        category_scores = features["technique_category_scores"]
        if category_scores:
            sorted_categories = sorted(
                category_scores.items(), 
                key=lambda x: x[1]["score"], 
                reverse=True
            )
            
            if sorted_categories[0][1]["score"] > 0:
                classification["primary_category"] = sorted_categories[0][0]
                classification["confidence_score"] = sorted_categories[0][1]["score"]
                
                # Add secondary categories with significant scores
                for category, data in sorted_categories[1:]:
                    if data["score"] >= 2:  # Threshold for secondary categories
                        classification["secondary_categories"].append(category)
                        
        # Determine content type
        content_type_scores = features["content_type_scores"]
        if content_type_scores:
            best_content_type = max(content_type_scores.items(), key=lambda x: x[1])
            if best_content_type[1] > 0:
                classification["content_type"] = best_content_type[0]
                
        # Subcategory classification
        if classification["primary_category"]:
            primary_cat = classification["primary_category"]
            subcategories = self.technique_taxonomy[primary_cat].get("subcategories", {})
            
            for subcat_name, subcat_keywords in subcategories.items():
                score = sum(1 for keyword in subcat_keywords if keyword in content.lower())
                if score > 0:
                    classification["subcategory_classifications"][subcat_name] = score
                    
        # Quality indicators
        classification["quality_indicators"] = {
            "has_structure": len(features["structure_indicators"]) > 0,
            "adequate_length": 200 <= features["word_count"] <= 2000,
            "good_readability": 30 <= features["readability_score"] <= 80,
            "professional_content": features["professional_term_density"] > 1.0,
            "actionable_content": classification["content_type"] in ["step_by_step_guide", "worksheet", "technique_description"]
        }
        
        return classification
        
    def enhance_metadata(self, item: Dict) -> Dict:
        """Enhance item metadata with processing information"""
        enhanced_item = item.copy()
        
        # Add processing timestamp
        enhanced_item["processing_timestamp"] = datetime.now().isoformat()
        
        # Perform intelligent classification
        classification = self.intelligent_classification(item)
        enhanced_item["classification"] = classification
        
        # Extract structured information
        structured_info = self.extract_structured_information(item.get("content", ""))
        enhanced_item["structured_information"] = structured_info
        
        # Calculate final quality score
        quality_score = self.calculate_enhanced_quality_score(item, classification)
        enhanced_item["enhanced_quality_score"] = quality_score
        
        return enhanced_item
        
    def extract_structured_information(self, content: str) -> Dict:
        """Extract structured information from content"""
        structured = {
            "techniques": [],
            "steps": [],
            "assessments": [],
            "key_points": [],
            "examples": [],
            "references": []
        }
        
        # Extract numbered steps
        step_patterns = [
            r"(?:step|stage) (\d+)[:.]?\s*(.+?)(?=(?:step|stage) \d+|$)",
            r"(\d+)[\.\)]\s*(.+?)(?=\d+[\.\)]|$)"
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) == 2:
                    step_info = {
                        "number": match[0],
                        "description": match[1].strip()[:200]  # Limit length
                    }
                    structured["steps"].append(step_info)
                    
        # Extract bullet points as key points
        bullet_patterns = [
            r"[•▪▫‣⁃]\s*(.+?)(?=[•▪▫‣⁃]|$)",
            r"[-*]\s*(.+?)(?=[-*]|$)"
        ]
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.strip()) > 10:  # Filter out very short items
                    structured["key_points"].append(match.strip()[:150])
                    
        # Extract examples
        example_patterns = [
            r"(?:for example|e\.g\.|such as|example:)\s*(.+?)(?=\.|$)",
            r"(?:case study|example)[:.]?\s*(.+?)(?=\n\n|$)"
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                structured["examples"].append(match.strip()[:200])
                
        return structured
        
    def calculate_enhanced_quality_score(self, item: Dict, classification: Dict) -> float:
        """Calculate enhanced quality score based on multiple factors"""
        score = 0.0
        
        # Base quality score from original assessment
        base_score = item.get("quality_assessment", {}).get("score", 0)
        score += base_score * 0.4  # 40% weight
        
        # Classification confidence
        confidence = classification.get("confidence_score", 0)
        score += min(confidence * 5, 20)  # Up to 20 points for classification confidence
        
        # Content type bonus
        content_type = classification.get("content_type")
        content_type_bonus = {
            "step_by_step_guide": 15,
            "technique_description": 12,
            "worksheet": 10,
            "assessment_tool": 8,
            "case_study": 6
        }
        score += content_type_bonus.get(content_type, 0)
        
        # Quality indicators bonus
        quality_indicators = classification.get("quality_indicators", {})
        for indicator, value in quality_indicators.items():
            if value:
                score += 5  # 5 points per quality indicator
                
        # Structure bonus
        structured_info = item.get("structured_information", {})
        if structured_info.get("steps"):
            score += 8  # Bonus for having structured steps
        if structured_info.get("key_points"):
            score += 5  # Bonus for having key points
            
        # Source credibility bonus
        source_type = item.get("source_type", "")
        source_bonus = {
            "government": 10,
            "clinical_guidelines": 8,
            "international_org": 8,
            "professional": 6,
            "academic": 5,
            "nonprofit": 3
        }
        score += source_bonus.get(source_type, 0)
        
        return min(score, 100)  # Cap at 100
        
    def process_all_enhanced_data(self) -> str:
        """Process all data with enhanced features"""
        self.logger.info("Starting enhanced data processing")
        
        # Load raw data
        raw_data = self.load_enhanced_raw_data()
        
        if not raw_data:
            self.logger.warning("No raw data found")
            return None
            
        # Advanced deduplication
        deduplicated_data = self.advanced_deduplication(raw_data)
        
        # Enhance metadata for each item
        enhanced_data = []
        for item in deduplicated_data:
            try:
                enhanced_item = self.enhance_metadata(item)
                enhanced_data.append(enhanced_item)
            except Exception as e:
                self.logger.error(f"Failed to enhance item: {e}")
                
        # Filter by enhanced quality score (use configured minimum)
        min_enhanced_score = self.load_min_enhanced_score()
        high_quality_data = [
            item for item in enhanced_data 
            if item.get("enhanced_quality_score", 0) >= min_enhanced_score
        ]
        
        self.logger.info(f"Quality filtering: {len(enhanced_data)} -> {len(high_quality_data)} items")
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.base_dir / "raw_data" / "processed" / f"cbt_enhanced_processed_{timestamp}.json"
        
        processing_metadata = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_raw_items": len(raw_data),
            "deduplicated_items": len(deduplicated_data),
            "final_items": len(high_quality_data),
            "average_quality_score": sum(item.get("enhanced_quality_score", 0) for item in high_quality_data) / len(high_quality_data) if high_quality_data else 0,
            "category_distribution": self.get_category_distribution(high_quality_data),
            "content_type_distribution": self.get_content_type_distribution(high_quality_data)
        }
        
        output_data = {
            "metadata": processing_metadata,
            "data": high_quality_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Enhanced processing completed: {output_file}")
        return str(output_file)
        
    def get_category_distribution(self, data: List[Dict]) -> Dict:
        """Get distribution of categories in processed data"""
        distribution = defaultdict(int)
        
        for item in data:
            primary_category = item.get("classification", {}).get("primary_category")
            if primary_category:
                distribution[primary_category] += 1
                
        return dict(distribution)
        
    def get_content_type_distribution(self, data: List[Dict]) -> Dict:
        """Get distribution of content types in processed data"""
        distribution = defaultdict(int)
        
        for item in data:
            content_type = item.get("classification", {}).get("content_type")
            if content_type:
                distribution[content_type] += 1
                
        return dict(distribution)

if __name__ == "__main__":
    processor = EnhancedCBTDataProcessor()
    
    print("Starting enhanced CBT data processing...")
    result = processor.process_all_enhanced_data()
    
    if result:
        print(f"Enhanced processing completed successfully: {result}")
    else:
        print("Enhanced processing failed or no data found") 