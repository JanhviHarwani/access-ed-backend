# src/scraping_scripts/process_urls.py

from web_processor import WebProcessor
import os
import time
from typing import List

def read_urls(filename: str) -> List[str]:
    """Read URLs from a file, skipping comments and empty lines."""
    urls = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    return urls

def main():
    # Initialize the web processor
    processor = WebProcessor()
    
    # Get the path to the URLs file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urls_file = os.path.join(base_dir, 'data', 'urls.txt')
    
    # Check if URLs file exists
    if not os.path.exists(urls_file):
        print(f"Error: URLs file not found at {urls_file}")
        return
    
    # Read URLs
    urls = read_urls(urls_file)
    print(f"Found {len(urls)} URLs to process")
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        print(f"\nProcessing URL {i}/{len(urls)}")
        content_dict = processor.process_url(url)
        
        if content_dict:
            # Save raw content
            processor.save_content(content_dict, raw=True)
            
            # Optional: Save processed content
            processor.save_content(content_dict, raw=False)
        
        # Be nice to servers
        if i < len(urls):
            time.sleep(2)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()