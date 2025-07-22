#!/usr/bin/env python3
"""
CBT Data Collection System
Collects CBT resources from public domain sources
"""

import requests
import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging

class CBTDataCollector:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
        
        # Public domain CBT sources
        self.sources = {
            "nhs_cbt": {
                "base_url": "https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/cognitive-behavioural-therapy-cbt/",
                "type": "government",
                "license": "open_government_license"
            },
            "cci_australia": {
                "base_url": "https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself",
                "type": "government", 
                "license": "creative_commons"
            },
            "nimh_psychotherapy": {
                "base_url": "https://www.nimh.nih.gov/health/topics/psychotherapies",
                "type": "government",
                "license": "public_domain"
            }
        }
        
    def setup_directories(self):
        """Create directory structure for CBT resources"""
        directories = [
            "raw_data/government",
            "raw_data/academic", 
            "raw_data/processed",
            "structured_data/techniques",
            "structured_data/assessments",
            "structured_data/worksheets",
            "structured_data/protocols",
            "embeddings"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_dir / 'collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def collect_web_content(self, url, source_name, max_pages=10):
        """Collect content from web sources"""
        self.logger.info(f"Starting collection from {source_name}: {url}")
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'CBT-Research-Tool/1.0 (Educational-Purpose)'
        })
        
        collected_data = []
        
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            content_data = self.extract_content(soup, url, source_name)
            if content_data:
                collected_data.append(content_data)
                
            # Find and process related links
            related_links = self.find_related_cbt_links(soup, url)
            
            for i, link in enumerate(related_links[:max_pages-1]):
                try:
                    time.sleep(2)  # Rate limiting
                    
                    link_response = session.get(link, timeout=30)
                    link_response.raise_for_status()
                    
                    link_soup = BeautifulSoup(link_response.content, 'html.parser')
                    link_content = self.extract_content(link_soup, link, source_name)
                    
                    if link_content:
                        collected_data.append(link_content)
                        
                    self.logger.info(f"Collected page {i+2}: {link}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect {link}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to collect from {url}: {e}")
            
        # Save collected data
        self.save_raw_data(collected_data, source_name)
        return collected_data
        
    def extract_content(self, soup, url, source_name):
        """Extract relevant content from HTML"""
        # Remove navigation, ads, scripts
        for element in soup(['nav', 'script', 'style', 'aside', 'header', 'footer']):
            element.decompose()
            
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text().strip()
            
        # Extract main content
        content_selectors = [
            'main',
            '[role="main"]',
            '.main-content',
            '.content',
            'article',
            '.article-body'
        ]
        
        main_content = ""
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                main_content = content_element.get_text(separator='\n', strip=True)
                break
                
        if not main_content:
            # Fallback to body content
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator='\n', strip=True)
                
        # Extract headings for structure
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            headings.append({
                'level': int(heading.name[1]),
                'text': heading.get_text().strip()
            })
            
        return {
            'url': url,
            'source': source_name,
            'title': title,
            'content': main_content,
            'headings': headings,
            'collected_at': time.time(),
            'word_count': len(main_content.split())
        }
        
    def find_related_cbt_links(self, soup, base_url):
        """Find CBT-related links on the page"""
        cbt_keywords = [
            'cognitive', 'behavioral', 'behaviour', 'therapy', 'cbt',
            'depression', 'anxiety', 'panic', 'phobia', 'stress',
            'mental-health', 'psychological', 'treatment'
        ]
        
        related_links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text().lower()
            
            # Convert relative to absolute URLs
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)
            
            # Only same domain links
            if parsed_url.netloc != base_domain:
                continue
                
            # Check if link is CBT-related
            url_lower = full_url.lower()
            text_lower = link_text.lower()
            
            is_cbt_related = any(
                keyword in url_lower or keyword in text_lower 
                for keyword in cbt_keywords
            )
            
            if is_cbt_related and full_url not in related_links:
                related_links.append(full_url)
                
        return related_links[:20]  # Limit to prevent excessive crawling
        
    def save_raw_data(self, data, source_name):
        """Save raw collected data"""
        timestamp = int(time.time())
        filename = f"{source_name}_{timestamp}.json"
        filepath = self.base_dir / "raw_data" / "government" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved {len(data)} items to {filepath}")
        
    def collect_all_sources(self):
        """Collect from all configured sources"""
        all_collected = {}
        
        for source_name, source_config in self.sources.items():
            self.logger.info(f"Starting collection for {source_name}")
            
            try:
                collected = self.collect_web_content(
                    source_config["base_url"], 
                    source_name,
                    max_pages=5  # Conservative limit
                )
                all_collected[source_name] = collected
                
            except Exception as e:
                self.logger.error(f"Collection failed for {source_name}: {e}")
                all_collected[source_name] = []
                
            time.sleep(5)  # Pause between sources
            
        return all_collected

def main():
    """Main collection function"""
    print("CBT Data Collection System")
    print("=" * 40)
    
    collector = CBTDataCollector()
    
    try:
        results = collector.collect_all_sources()
        
        total_items = sum(len(items) for items in results.values())
        print(f"\nCollection completed successfully!")
        print(f"Total items collected: {total_items}")
        
        for source, items in results.items():
            print(f"  {source}: {len(items)} items")
            
    except Exception as e:
        print(f"Collection failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    main() 