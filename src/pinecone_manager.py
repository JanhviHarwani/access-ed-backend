# src/pinecone_manager.py

from pinecone import Pinecone, ServerlessSpec
# from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Dict, List
import os
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)
class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = "accessibility-index"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cpu'}
        )
        
        self._init_index()

    def _init_index(self):
        try:
            logger.info("Checking if index exists...")
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info("Index created successfully")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
            self.index = self.pc.Index(self.index_name)
            logger.info("Index connection established")
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}", exc_info=True)
            raise

    async def add_documents(self, documents: List[Dict]):
        vectors = []
        
        logger.info(f"Preparing to insert {len(documents)} documents into Pinecone...")

        for i, doc in enumerate(documents):
            embedding = self.embeddings.embed_query(doc['content'])
            vectors.append({
                'id': f"doc_{i}",
                'values': embedding,
                'metadata': {
                    'content': doc['content'],
                    'title': doc.get('title', ''),
                    'category': doc.get('category', ''),
                    'source': doc.get('source', ''),
                    'source_url': doc.get('source_url', '') 
                }
            })
            if i % 50 == 0:
                logger.info(f"Processed {i} documents...")

        logger.info(f"Prepared {len(vectors)} vectors for Pinecone.")

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            logger.info(f"Uploading batch {i // batch_size + 1} of {len(vectors) // batch_size + 1} to Pinecone...")
            
            try:
                upsert_response = self.index.upsert(vectors=batch)
                logger.info(f"Successfully uploaded batch with response: {upsert_response}")
            except Exception as e:
                logger.error(f"Error uploading batch {i // batch_size + 1}: {str(e)}")

    def search(self, query: str, category: str = None, top_k: int = 3):
        query_embedding = self.embeddings.embed_query(query)
        
        filter_dict = {"category": {"$eq": category}} if category else None
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        return results
    def delete_all_vectors(self):
        try:
            # Check if the index exists before deleting
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.error(f"Index '{self.index_name}' not found. Skipping deletion.")
                return
            
            # Check if there are any namespaces before attempting to delete
            index_stats = self.index.describe_index_stats()
            namespaces = index_stats.get("namespaces", {})

            if not namespaces:
                logger.warning("No namespaces found in index. Skipping vector deletion.")
                return

            # Proceed with deletion if namespaces exist
            self.index.delete(deleteAll=True)
            logger.info("Deleted all vectors from the index successfully.")
        
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")

