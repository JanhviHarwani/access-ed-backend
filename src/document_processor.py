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
            
            # Process each category
            for category in os.listdir(self.categories_dir):
                category_path = os.path.join(self.categories_dir, category)
                if os.path.isdir(category_path):
                    logger.info(f"Processing category: {category}")
                    
                    # Process each file in category
                    for filename in os.listdir(category_path):
                        if filename.endswith('.txt'):
                            filepath = os.path.join(category_path, filename)
                            chunks = self._process_file(filepath, category, filename)
                            documents.extend(chunks)
            
            logger.info(f"Processed {len(documents)} total chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
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
            
            # Extract content and create base metadata
            content = file_content[0].page_content
            metadata = {
                'category': category,
                'filename': filename,
                'source': filepath,
                'original_size': len(content)
            }
            
            # Chunk the content
            chunks = self.chunker.chunk_document(content, metadata)
            
            # Convert chunks to dictionary format
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'content': chunk.content,
                    **chunk.metadata  # Include all metadata
                }
                chunk_dicts.append(chunk_dict)
            
            logger.info(f"Created {len(chunk_dicts)} chunks from {filename}")
            
            # Log chunk statistics
            stats = self.chunker.get_chunk_info(chunks)
            logger.info(f"Chunk statistics for {filename}: {stats}")
            
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