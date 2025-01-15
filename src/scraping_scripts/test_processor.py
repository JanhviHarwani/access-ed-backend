# src/scraping_scripts/test_processor.py

from web_processor import ContentProcessor

def test_single_url():
    processor = ContentProcessor()
    
    # Test with a single URL
    test_url = "https://www.who.int/health-topics/blindness-and-vision-loss"
    
    print("Testing with single URL...")
    doc = processor.process_url(test_url)
    
    if doc:
        print("\nExtracted content:")
        print(f"Title: {doc['title']}")
        print(f"Content length: {len(doc['content'])} characters")
        
        # Save the document
        processor.save_document(doc)
        print("\nTest completed successfully!")
    else:
        print("Failed to process URL")

if __name__ == "__main__":
    test_single_url()