# src/scraping_scripts/web_processor.py

import requests
from bs4 import BeautifulSoup
import os
import time
import ssl
import certifi
from typing import Dict, Optional
from urllib.parse import urlparse
import logging
from content_formatter import ContentFormatter
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebProcessor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, 'data', 'processed')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Setup session with custom SSL context
        self.session = requests.Session()
        self.session.verify = certifi.where()
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logger.info(f"Initialized WebProcessor. Data directory: {self.data_dir}")

    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        try:
            # Split into lines
            lines = text.split('\n')
            
            # Clean each line
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Skip empty lines and common unwanted content
                if line and not any(skip in line.lower() for skip in ['cookie', 'privacy policy', 'terms of use']):
                    cleaned_lines.append(line)
            
            # Join lines with proper spacing
            return '\n'.join(cleaned_lines)
        except Exception as e:
            logger.error(f"Error in clean_text: {e}")
            return text

    def extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        # Remove unwanted elements
        for elem in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            elem.decompose()
        
        # Try to find main content area
        content_areas = []
        
        # Look for common content containers
        for tag in ['main', 'article', 'section']:
            elements = soup.find_all(tag)
            content_areas.extend(elements)
        
        # Look for divs with content-related classes
        content_classes = ['content', 'main-content', 'article-content', 'post-content']
        for class_name in content_classes:
            elements = soup.find_all('div', class_=class_name)
            content_areas.extend(elements)
        
        if content_areas:
            # Combine content from all found areas
            content = '\n'.join(area.get_text(separator='\n') for area in content_areas)
        else:
            # Fallback to body content
            content = soup.find('body').get_text(separator='\n') if soup.find('body') else ''
        
        return self.clean_text(content)

    def process_url(self, url: str) -> Optional[Dict]:
        """Process a single URL."""
        try:
            logger.info(f"Processing URL: {url}")
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=15,
                verify=certifi.where()
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else ''
            
            # Extract content
            content = self.extract_content(soup)
            
            if not content.strip():
                logger.warning(f"No content extracted from {url}")
                return None
            
            doc = {
                'url': url,
                'title': title.strip() if title else '',
                'content': content,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Successfully processed URL: {url}")
            return doc
            
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL Error for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            return None
    
    def save_document(self, doc: Dict):
        """Save processed document with better formatting.
        
        The filename will be the document's title converted to lowercase with spaces
        replaced by hyphens, e.g., 'eye-care-vision-impairment-and-blindness.txt'
        """
        if not doc:
            return
            
        try:
            formatter = ContentFormatter()
            formatted_doc = formatter.format_content(doc)
            
            # Clean the title for use in filename
            title = doc.get('title', '').strip()
            if not title:
                title = 'untitled'
                
            # Convert to lowercase first
            title = title.lower()
            
            # Replace any punctuation or special characters with spaces
            clean_title = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in title)
            
            # Replace one or more spaces with a single hyphen
            clean_title = '-'.join(word for word in clean_title.split())
            
            filename = f"{clean_title}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            # Handle duplicate filenames by adding a number if needed
            base_path = os.path.splitext(filepath)[0]
            counter = 1
            while os.path.exists(filepath):
                filepath = f"{base_path}-{counter}.txt"
                counter += 1
            
            formatter.save_formatted_content(formatted_doc, filepath)
            logger.info(f"Saved formatted document to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            raise
    def process_urls_from_file(self, urls_file: str) -> None:
        """Process URLs from a file and log successes and failures."""
        try:
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Found {len(urls)} URLs to process")
            
            processed_count = 0
            failed_urls = []
            
            for url in urls:
                doc = self.process_url(url)
                if doc:
                    try:
                        self.save_document(doc)
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error saving document for URL {url}: {e}")
                        failed_urls.append(url)
                else:
                    failed_urls.append(url)
                
                time.sleep(2)  # Be nice to servers
            
            logger.info(f"Processing complete! Successfully processed {processed_count} URLs")
            if failed_urls:
                logger.warning(f"Failed to process {len(failed_urls)} URLs:")
                for failed_url in failed_urls:
                    logger.warning(f"  - {failed_url}")
            
        except Exception as e:
            logger.error(f"Error processing URLs from file: {e}")


def main():
    try:
        processor = WebProcessor()
        
        # Get the path to the URLs file
        urls_file = os.path.join(processor.base_dir, 'data', 'urls.txt')
        
        if not os.path.exists(urls_file):
            logger.error(f"URLs file not found at {urls_file}")
            return
            
        processed_count = processor.process_urls_from_file(urls_file)
        logger.info(f"Processing complete! Successfully processed {processed_count} URLs")
        
    except Exception as e:
        logger.error(f"Main process error: {e}")

if __name__ == "__main__":
    main()