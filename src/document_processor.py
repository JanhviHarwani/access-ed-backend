# src/document_processor.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Dict
from pinecone_manager import PineconeManager  # Change this line

class DocumentProcessor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.categories_dir = os.path.join(self.base_dir, 'data', 'categories')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.pinecone_manager = PineconeManager()
    def process_documents(self) -> List[Dict]:
        RELOAD_DOCUMENTS = os.getenv('RELOAD_DOCUMENTS', 'false').lower() == 'true'
        
        if not RELOAD_DOCUMENTS:
            print("Skipping document processing - RELOAD_DOCUMENTS is false")
            return []
        
        # Everything below this point only runs if RELOAD_DOCUMENTS is true
        print("Starting document processing...")
        
        # Step 1: Clean up existing index
        print("Cleaning up Pinecone index...")
        self.pinecone_manager.delete_all_vectors()  
        
        # Step 2: Process new documents
        documents = []
        print(f"Found categories: {os.listdir(self.categories_dir)}")
        
        for category in os.listdir(self.categories_dir):
            category_path = os.path.join(self.categories_dir, category)
            if os.path.isdir(category_path):
                print(f"Processing category: {category}")
                num_files = len([f for f in os.listdir(category_path) if f.endswith('.txt')])
                print(f"Found {num_files} files in {category}")
                
                for filename in os.listdir(category_path):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(category_path, filename)
                        loader = TextLoader(filepath, encoding='utf-8')
                        file_docs = loader.load()
                        
                        texts = self.text_splitter.split_documents(file_docs)
                        
                        for text in texts:
                            documents.append({
                                'content': text.page_content,
                                'category': category,
                                'filename': filename,
                                'source': filepath
                            })
        
        print(f"Processed {len(documents)} document chunks")
        return documents