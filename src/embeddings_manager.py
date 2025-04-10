from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self):
        load_dotenv()
        
        # Initialize Pinecone with new client
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = "accessibility-index"
        
        # Initialize embeddings model
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
            self.index = self.pc.Index(self.index_name)
            logger.info("Index connection established")
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}", exc_info=True)
            raise

    async def create_vector_store(self, texts):
        """Create or update vector store with document chunks."""
        vectors = []
        
        for i, doc in enumerate(texts):
            embedding = self.embeddings.embed_query(doc.page_content)
            vectors.append({
                'id': f"doc_{i}",
                'values': embedding,
                'metadata': {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
            })
            
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                logger.info(f"Successfully uploaded batch {i // batch_size + 1}")
            except Exception as e:
                logger.error(f"Error uploading batch: {str(e)}")

    def get_relevant_chunks(self, query, k=3):
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        return results