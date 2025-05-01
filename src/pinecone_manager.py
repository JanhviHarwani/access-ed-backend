from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import os
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
                logger.info(f"Index '{self.index_name}' not found")
                raise Exception(f"Index '{self.index_name}' does not exist")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
            self.index = self.pc.Index(self.index_name)
            logger.info("Index connection established")
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}", exc_info=True)
            raise

    def search(self, query: str, category: str = None, top_k: int = 3):
        """Search for relevant documents in Pinecone"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            filter_dict = {"category": {"$eq": category}} if category else None
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return None 