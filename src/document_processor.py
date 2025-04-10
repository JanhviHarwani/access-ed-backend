import re
from langchain_community.document_loaders import TextLoader
import os
from typing import List, Dict
from auto_chunker import AutoChunker  # The class we just created
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.categories_dir = os.path.join(self.base_dir, 'data', 'categories')
        self.chunker = AutoChunker(
            max_chunk_size=500,  # Maximum size of each chunk
            min_chunk_size=100,  # Minimum size for independent chunks
            overlap_size=50      # Overlap between chunks
        )

    def process_documents(self) -> List[Dict]:
        try:
            logger.info("Starting document processing...")
            documents = []
            
            # Get all categories first and log them
            categories = os.listdir(self.categories_dir)
            logger.info(f"Found categories: {categories}")
            
            # Process each category
            for category in categories:
                category_path = os.path.join(self.categories_dir, category)
                if os.path.isdir(category_path):
                    logger.info(f"Processing category: {category}")
                    
                    # Get and log all files in category first
                    files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
                    logger.info(f"Found {len(files)} files in category {category}: {files}")
                    
                    # Process each file in category
                    for filename in files:
                        try:
                            filepath = os.path.join(category_path, filename)
                            logger.info(f"Starting to process file: {filepath}")
                            chunks = self._process_file(filepath, category, filename)
                            documents.extend(chunks)
                            logger.info(f"Successfully processed {filename}")
                        except Exception as e:
                            logger.error(f"Error processing file {filename}: {str(e)}")
                            continue  # Continue with next file even if one fails
                    
                    logger.info(f"Finished processing category: {category}")
            
            logger.info(f"Processed {len(documents)} total chunks across all categories")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            raise

    def _process_file(self, filepath: str, category: str, filename: str) -> List[Dict]:
        """Process a single file into chunks"""
        try:
            # Load the file
            loader = TextLoader(filepath, encoding='utf-8')
            file_content = loader.load()
            
            if not file_content:
                logger.warning(f"Empty file: {filepath}")
                return []
            
            # Extract content and parse metadata
            content = file_content[0].page_content
            
            # Parse the metadata from content
            title = None
            source_url = None
            actual_content = content
            
            # Extract metadata from content
            title_match = re.search(r'Title: (.*?)\n', content)
            url_match = re.search(r'Source URL: (.*?)\n', content)
            content_start = content.find('Content:')
            
            if title_match:
                title = title_match.group(1).strip()
            if url_match:
                source_url = url_match.group(1).strip()
            if content_start != -1:
                actual_content = content[content_start + 8:].strip()  # +8 for "Content:"
            
            # Create base metadata
            metadata = {
                'category': category,
                'filename': filename,
                'source': filepath,
                'original_size': len(content),
                'title': title,
                'source_url': source_url
            }
            
            # Chunk only the actual content part
            chunks = self.chunker.chunk_document(actual_content, metadata)
            
            # Convert chunks to dictionary format while preserving all metadata
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'content': f"Title: {title}\nSource URL: {source_url}\n\nContent:\n{chunk.content}",
                    **chunk.metadata  # Include all metadata
                }
                chunk_dicts.append(chunk_dict)
            
            logger.info(f"Created {len(chunk_dicts)} chunks from {filename}")
            
            return chunk_dicts
                
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            return []
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about the document processing"""
        try:
            total_files = 0
            total_chunks = 0
            categories = {}
            
            for category in os.listdir(self.categories_dir):
                category_path = os.path.join(self.categories_dir, category)
                if os.path.isdir(category_path):
                    files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
                    categories[category] = len(files)
                    total_files += len(files)
            
            return {
                'total_files': total_files,
                'total_chunks': total_chunks,
                'categories': categories
            }
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {}