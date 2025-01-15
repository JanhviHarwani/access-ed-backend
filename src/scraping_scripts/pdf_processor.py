import os
import logging
from typing import Dict, Optional
import re
from datetime import datetime
from PyPDF2 import PdfReader
from pdf_content_formatter import ContentFormatter  # Assuming similar formatter class exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, 'data', 'processed')
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Initialized PDFProcessor. Data directory: {self.data_dir}")

    def clean_text(self, text: str) -> str:
        """Clean extracted text from PDF."""
        try:
            # Split into lines
            lines = text.split('\n')
            
            # Clean each line
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Skip empty lines and page numbers
                if line and not re.match(r'^Page \d+$', line):
                    cleaned_lines.append(line)
            
            # Join lines with proper spacing
            return '\n'.join(cleaned_lines)
        except Exception as e:
            logger.error(f"Error in clean_text: {e}")
            return text

    def extract_metadata(self, pdf_reader: PdfReader) -> Dict:
        """Extract metadata from PDF document."""
        try:
            metadata = {}
            if pdf_reader.metadata:
                metadata = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                    'producer': pdf_reader.metadata.get('/Producer', '')
                }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def extract_content(self, pdf_path: str) -> Optional[Dict]:
        """Extract content and metadata from PDF file."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Open PDF file
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Extract text from all pages
                content = []
                for page in pdf_reader.pages:
                    content.append(page.extract_text())
                
                # Combine and clean content
                full_content = self.clean_text('\n'.join(content))
                
                if not full_content.strip():
                    logger.warning(f"No content extracted from {pdf_path}")
                    return None
                
                # Get metadata
                metadata = self.extract_metadata(pdf_reader)
                
                # Create document dictionary
                doc = {
                    'file_path': pdf_path,
                    'title': metadata.get('title', os.path.basename(pdf_path)),
                    'content': full_content,
                    'metadata': metadata,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                logger.info(f"Successfully processed PDF: {pdf_path}")
                return doc
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None

    def save_document(self, doc: Dict):
        """Save processed document with formatting."""
        if not doc:
            return
            
        try:
            formatter = ContentFormatter()
            formatted_doc = formatter.format_content(doc)
            
            # Create filename from title or original filename
            title = doc.get('title', '').strip()
            if not title:
                title = os.path.splitext(os.path.basename(doc['file_path']))[0]
                
            # Clean filename
            title = title.lower()
            clean_title = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in title)
            filename = f"{'-'.join(clean_title.split())}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            # Handle duplicates
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

    def process_pdfs_from_directory(self, pdf_dir: str) -> int:
        """Process all PDFs from a directory."""
        try:
            pdf_files = [f for f in os.listdir(pdf_dir) 
                        if f.lower().endswith('.pdf')]
            
            logger.info(f"Found {len(pdf_files)} PDFs to process")
            
            processed_count = 0
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                doc = self.extract_content(pdf_path)
                if doc:
                    self.save_document(doc)
                    processed_count += 1
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing PDFs from directory: {e}")
            return 0

def main():
    try:
        processor = PDFProcessor()
        
        # Get the path to the PDFs directory
        pdfs_dir = os.path.join(processor.base_dir, 'data', 'pdfs')
        
        if not os.path.exists(pdfs_dir):
            logger.error(f"PDFs directory not found at {pdfs_dir}")
            return
            
        processed_count = processor.process_pdfs_from_directory(pdfs_dir)
        logger.info(f"Processing complete! Successfully processed {processed_count} PDFs")
        
    except Exception as e:
        logger.error(f"Main process error: {e}")

if __name__ == "__main__":
    main()