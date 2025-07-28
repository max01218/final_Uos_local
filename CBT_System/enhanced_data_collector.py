#!/usr/bin/env python3
"""
Enhanced CBT Data Collection System
Expanded sources including APA, WHO, NICE, and academic resources
"""

import requests
import json
import os
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

class EnhancedCBTDataCollector:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
        
        # Expanded public domain CBT sources
        self.sources = {
            # Government sources (existing)
            "nhs_cbt": {
                "base_url": "https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/cognitive-behavioural-therapy-cbt/",
                "type": "government",
                "license": "open_government_license",
                "priority": 1
            },
            "cci_australia": {
                "base_url": "https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself",
                "type": "government", 
                "license": "creative_commons",
                "priority": 1
            },
            "nimh_psychotherapy": {
                "base_url": "https://www.nimh.nih.gov/health/topics/psychotherapies",
                "type": "government",
                "license": "public_domain",
                "priority": 1
            },
            
            # Professional organizations
            "apa_therapy": {
                "base_url": "https://www.apa.org/topics/therapy/",
                "type": "professional",
                "license": "educational_use",
                "priority": 2
            },
            "apa_depression": {
                "base_url": "https://www.apa.org/depression-guideline/",
                "type": "professional",
                "license": "educational_use", 
                "priority": 2
            },
            
            # International health organizations
            "who_mental_health": {
                "base_url": "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
                "type": "international_org",
                "license": "creative_commons",
                "priority": 1
            },
            "who_self_help": {
                "base_url": "https://www.who.int/publications/i/item/9789240003927",
                "type": "international_org",
                "license": "creative_commons",
                "priority": 1
            },
            
            # UK NICE guidelines
            "nice_depression": {
                "base_url": "https://www.nice.org.uk/guidance/ng222",
                "type": "clinical_guidelines",
                "license": "open_government_license",
                "priority": 1
            },
            "nice_anxiety": {
                "base_url": "https://www.nice.org.uk/guidance/cg113",
                "type": "clinical_guidelines", 
                "license": "open_government_license",
                "priority": 1
            },
            
            # Canadian resources
            "camh_resources": {
                "base_url": "https://www.camh.ca/en/health-info/mental-illness-and-addiction-index",
                "type": "healthcare_org",
                "license": "educational_use",
                "priority": 2
            },
            
            # Academic and research
            "cochrane_cbt": {
                "base_url": "https://www.cochranelibrary.com/topic/behavioural-sciences",
                "type": "academic",
                "license": "open_access",
                "priority": 3
            },
            
            # Self-help and public resources
            "mental_health_america": {
                "base_url": "https://www.mhanational.org/conditions",
                "type": "nonprofit",
                "license": "educational_use",
                "priority": 2
            }
        }
        
        # Quality assessment criteria (load from config if available)
        self.quality_criteria = self.load_quality_criteria()
        
    def load_quality_criteria(self) -> Dict:
        """Load quality criteria from config file or use defaults"""
        config_file = "enhanced_config.json"
        default_criteria = {
            "min_content_length": 200,
            "max_content_length": 5000,
            "required_keywords": ["cbt", "cognitive", "behavioral", "therapy", "technique"],
            "exclude_keywords": ["advertisement", "commercial", "purchase", "buy"],
            "professional_terms_threshold": 3,
            "minimum_quality_score": 60
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                criteria = config.get("data_collection", {}).get("quality_criteria", {})
                
                # Map config keys to internal keys
                quality_criteria = {
                    "min_length": criteria.get("min_content_length", default_criteria["min_content_length"]),
                    "max_length": criteria.get("max_content_length", default_criteria["max_content_length"]),
                    "required_keywords": criteria.get("required_keywords", default_criteria["required_keywords"]),
                    "exclude_keywords": criteria.get("exclude_keywords", default_criteria["exclude_keywords"]),
                    "professional_terms_threshold": criteria.get("professional_terms_threshold", default_criteria["professional_terms_threshold"]),
                    "minimum_quality_score": criteria.get("minimum_quality_score", default_criteria["minimum_quality_score"])
                }
                
                return quality_criteria
            else:
                return default_criteria
                
        except Exception as e:
            print(f"Warning: Failed to load config, using defaults: {e}")
            return default_criteria
        
    def setup_directories(self):
        """Create enhanced directory structure"""
        directories = [
            "raw_data/government",
            "raw_data/professional_orgs",
            "raw_data/international_orgs", 
            "raw_data/clinical_guidelines",
            "raw_data/academic",
            "raw_data/nonprofit",
            "raw_data/processed",
            "structured_data/techniques",
            "structured_data/assessments",
            "structured_data/worksheets",
            "structured_data/protocols",
            "structured_data/self_help_tools",
            "embeddings",
            "quality_reports",
            "metadata"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Enhanced logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # File handler
        file_handler = logging.FileHandler(self.base_dir / 'enhanced_collection.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        
        # Setup logger
        self.logger = logging.getLogger("EnhancedCBTCollector")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
        
    def assess_content_quality(self, content: str, url: str = "") -> Dict:
        """Assess content quality based on multiple criteria"""
        assessment = {
            "score": 0,
            "length_ok": False,
            "has_required_keywords": False,
            "no_excluded_keywords": True,
            "professional_terms_count": 0,
            "readability_score": 0,
            "pass": False
        }
        
        content_lower = content.lower()
        
        # Length check
        if self.quality_criteria["min_length"] <= len(content) <= self.quality_criteria["max_length"]:
            assessment["length_ok"] = True
            assessment["score"] += 20
            
        # Required keywords check (more lenient)
        required_found = sum(1 for kw in self.quality_criteria["required_keywords"] if kw in content_lower)
        if required_found >= 1:
            assessment["has_required_keywords"] = True
            assessment["score"] += 30
            # Bonus for more keywords
            if required_found >= 3:
                assessment["score"] += 10
            
        # Excluded keywords check
        excluded_found = sum(1 for kw in self.quality_criteria["exclude_keywords"] if kw in content_lower)
        if excluded_found > 0:
            assessment["no_excluded_keywords"] = False
            assessment["score"] -= 20
            
        # Professional terms count
        professional_terms = [
            "intervention", "therapeutic", "evidence-based", "clinical", "research",
            "systematic", "randomized", "efficacy", "assessment", "diagnosis"
        ]
        prof_count = sum(1 for term in professional_terms if term in content_lower)
        assessment["professional_terms_count"] = prof_count
        
        if prof_count >= self.quality_criteria["professional_terms_threshold"]:
            assessment["score"] += 25
        elif prof_count >= 1:
            assessment["score"] += 15  # Partial credit for having some professional terms
            
        # Simple readability (sentence length variance)
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 3:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_length <= 25:  # Reasonable sentence length
                assessment["readability_score"] = 15
                assessment["score"] += 15
                
        # Overall pass threshold (use configured minimum)
        min_score = self.quality_criteria.get("minimum_quality_score", 60)
        assessment["pass"] = assessment["score"] >= min_score
        
        return assessment
        
    def extract_structured_content(self, content: str, source_type: str) -> Dict:
        """Extract and structure CBT-specific content"""
        structured = {
            "techniques": [],
            "assessments": [],
            "worksheets": [],
            "key_concepts": [],
            "step_by_step_guides": []
        }
        
        content_lower = content.lower()
        
        # Extract techniques
        technique_patterns = [
            r"(\d+[\.\)].*?(?:technique|method|approach|strategy).*?)(?=\d+[\.\)]|\n\n|$)",
            r"(step \d+:.*?)(?=step \d+:|$)",
            r"(exercise \d+:.*?)(?=exercise \d+:|$)"
        ]
        
        for pattern in technique_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            structured["techniques"].extend(matches)
            
        # Extract key concepts
        concept_patterns = [
            r"(cognitive distortion[s]?.*?)(?=\n\n|\. [A-Z])",
            r"(automatic thought[s]?.*?)(?=\n\n|\. [A-Z])",
            r"(behavioral activation.*?)(?=\n\n|\. [A-Z])"
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            structured["key_concepts"].extend(matches)
            
        return structured
        
    def collect_enhanced_content(self, url: str, source_name: str, source_config: Dict, depth: int = 0, max_depth: int = 2) -> List[Dict]:
        """Enhanced content collection with quality assessment"""
        # Prevent infinite recursion
        if depth > max_depth:
            self.logger.debug(f"Max depth {max_depth} reached for {url}")
            return []
            
        self.logger.info(f"Starting enhanced collection from {source_name}: {url} (depth: {depth})")
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'CBT-Research-Tool/2.0 (Educational-Research-Purpose)'
        })
        
        collected_data = []
        processed_urls = set()
        
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            # Handle encoding properly
            if response.encoding is None:
                response.encoding = response.apparent_encoding or 'utf-8'
            
            # Use response.text instead of response.content for better encoding handling
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content_selectors = [
                'article', 'main', '.content', '#content', 
                '.main-content', '.article-content', '.page-content'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
                    
            if not content_element:
                content_element = soup.find('body')
                
            if content_element:
                # Clean content
                for tag in content_element(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    tag.decompose()
                    
                content_text = content_element.get_text(separator=' ', strip=True)
                content_text = re.sub(r'\s+', ' ', content_text)
                
                # Quality assessment
                quality_assessment = self.assess_content_quality(content_text, url)
                
                if quality_assessment["pass"]:
                    # Extract structured content
                    structured_content = self.extract_structured_content(content_text, source_config["type"])
                    
                    # Create data entry
                    data_entry = {
                        "url": url,
                        "source": source_name,
                        "source_type": source_config["type"],
                        "license": source_config["license"],
                        "priority": source_config["priority"],
                        "content": content_text,
                        "content_hash": self.calculate_content_hash(content_text),
                        "structured_content": structured_content,
                        "quality_assessment": quality_assessment,
                        "collection_timestamp": datetime.now().isoformat(),
                        "content_length": len(content_text),
                        "title": soup.find('title').get_text() if soup.find('title') else ""
                    }
                    
                    collected_data.append(data_entry)
                    self.logger.info(f"Collected high-quality content from {url} (Score: {quality_assessment['score']})")
                    
                else:
                    self.logger.warning(f"Content from {url} failed quality assessment (Score: {quality_assessment['score']})")
                    
            # Look for additional relevant links
            if len(collected_data) < 5:  # Collect more if needed
                links = soup.find_all('a', href=True)
                relevant_links = []
                
                for link in links:
                    href = link.get('href')
                    if href and not href.startswith('#'):
                        full_url = urljoin(url, href)
                        
                        if full_url not in processed_urls and self.is_relevant_link(link.get_text(), href):
                            relevant_links.append(full_url)
                            
                # Collect from relevant links (limit to prevent excessive requests)
                for link_url in relevant_links[:3]:
                    if link_url not in processed_urls:
                        processed_urls.add(link_url)
                        time.sleep(2)  # Rate limiting
                        
                        try:
                            link_data = self.collect_enhanced_content(link_url, source_name, source_config, depth + 1, max_depth)
                            collected_data.extend(link_data)
                        except Exception as e:
                            self.logger.warning(f"Failed to collect from link {link_url}: {e}")
                            
        except Exception as e:
            self.logger.error(f"Failed to collect from {url}: {e}")
            
        return collected_data
        
    def is_relevant_link(self, link_text: str, href: str) -> bool:
        """Check if a link is relevant to CBT content"""
        
        # Exclude social media and sharing links
        excluded_domains = [
            "facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com",
            "youtube.com", "tiktok.com", "pinterest.com", "reddit.com",
            "mailto:", "tel:", "javascript:", "#"
        ]
        
        # Exclude common non-content patterns
        excluded_patterns = [
            "share", "print", "email", "download", "pdf", "login", "signup",
            "subscribe", "newsletter", "cookie", "privacy", "terms"
        ]
        
        href_lower = href.lower()
        text_lower = link_text.lower()
        
        # Filter out excluded domains and patterns
        if any(domain in href_lower for domain in excluded_domains):
            return False
            
        if any(pattern in href_lower or pattern in text_lower for pattern in excluded_patterns):
            return False
            
        # Check for relevant content keywords
        relevant_keywords = [
            "cbt", "cognitive", "behavioral", "therapy", "treatment",
            "depression", "anxiety", "technique", "intervention", "mental", 
            "health", "counselling", "psychological", "psychotherapy"
        ]
        
        return any(keyword in text_lower or keyword in href_lower for keyword in relevant_keywords)
        
    def save_enhanced_data(self, data: List[Dict], source_name: str, source_type: str) -> str:
        """Save collected data with enhanced metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}_{timestamp}.json"
        
        # Determine directory based on source type
        type_dir_map = {
            "government": "government",
            "professional": "professional_orgs",
            "international_org": "international_orgs",
            "clinical_guidelines": "clinical_guidelines",
            "academic": "academic",
            "nonprofit": "nonprofit"
        }
        
        directory = type_dir_map.get(source_type, "government")
        file_path = self.base_dir / "raw_data" / directory / filename
        
        # Add collection metadata
        collection_metadata = {
            "collection_timestamp": datetime.now().isoformat(),
            "source_name": source_name,
            "source_type": source_type,
            "total_items": len(data),
            "quality_scores": [item["quality_assessment"]["score"] for item in data],
            "average_quality": sum(item["quality_assessment"]["score"] for item in data) / len(data) if data else 0
        }
        
        output_data = {
            "metadata": collection_metadata,
            "data": data
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved {len(data)} items to {file_path}")
        return str(file_path)
        
    def collect_from_source(self, source_name: str) -> List[Dict]:
        """Collect data from a specific enhanced source"""
        if source_name not in self.sources:
            self.logger.error(f"Unknown source: {source_name}")
            return []
            
        source_config = self.sources[source_name]
        self.logger.info(f"Collecting from {source_name} (Priority: {source_config['priority']})")
        
        collected_data = self.collect_enhanced_content(
            source_config["base_url"], 
            source_name, 
            source_config
        )
        
        if collected_data:
            self.save_enhanced_data(collected_data, source_name, source_config["type"])
            
        return collected_data
        
    def collect_all_enhanced_sources(self, priority_filter: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Collect from all enhanced sources with optional priority filtering"""
        results = {}
        
        # Sort sources by priority
        sorted_sources = sorted(
            self.sources.items(), 
            key=lambda x: x[1]["priority"]
        )
        
        for source_name, source_config in sorted_sources:
            if priority_filter is None or source_config["priority"] <= priority_filter:
                try:
                    data = self.collect_from_source(source_name)
                    results[source_name] = data
                    
                    # Rate limiting between sources
                    time.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"Failed to collect from {source_name}: {e}")
                    results[source_name] = []
                    
        # Generate collection report
        self.generate_collection_report(results)
        
        return results
        
    def generate_collection_report(self, results: Dict[str, List[Dict]]) -> None:
        """Generate detailed collection report"""
        report = {
            "collection_timestamp": datetime.now().isoformat(),
            "total_sources": len(results),
            "successful_sources": len([k for k, v in results.items() if v]),
            "total_items": sum(len(items) for items in results.values()),
            "source_breakdown": {},
            "quality_summary": {
                "high_quality": 0,  # Score >= 80
                "medium_quality": 0,  # Score 60-79
                "low_quality": 0   # Score < 60
            }
        }
        
        for source_name, items in results.items():
            source_stats = {
                "items_collected": len(items),
                "average_quality": 0,
                "content_types": set()
            }
            
            if items:
                scores = [item["quality_assessment"]["score"] for item in items]
                source_stats["average_quality"] = sum(scores) / len(scores)
                
                for item in items:
                    score = item["quality_assessment"]["score"]
                    if score >= 80:
                        report["quality_summary"]["high_quality"] += 1
                    elif score >= 60:
                        report["quality_summary"]["medium_quality"] += 1
                    else:
                        report["quality_summary"]["low_quality"] += 1
                        
                    source_stats["content_types"].add(item["source_type"])
                    
            source_stats["content_types"] = list(source_stats["content_types"])
            report["source_breakdown"][source_name] = source_stats
            
        # Save report
        report_path = self.base_dir / "quality_reports" / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Collection report saved to {report_path}")
        self.logger.info(f"Collection summary: {report['total_items']} items from {report['successful_sources']}/{report['total_sources']} sources")

if __name__ == "__main__":
    collector = EnhancedCBTDataCollector()
    
    # Collect from priority 1 sources first (government and international orgs)
    print("Collecting from priority 1 sources...")
    results = collector.collect_all_enhanced_sources(priority_filter=1)
    
    print(f"\nCollection completed: {sum(len(items) for items in results.values())} total items") 