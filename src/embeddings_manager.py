# src/embeddings_manager.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv

class EmbeddingsManager:
    def __init__(self):
        load_dotenv()
        
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENV')
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cpu'}  # Change to 'cuda' if using GPU
        )
        
        self.index_name = "accessibility-index"

    def create_vector_store(self, texts):
        """Create or update vector store with document chunks."""
        # Create vector store if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=768,  # BGE-base-en dimension
                metric="cosine"
            )

        # Create vector store
        vector_store = Pinecone.from_documents(
            documents=texts,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        
        return vector_store

    def get_relevant_chunks(self, query, k=3):
        """Retrieve relevant chunks for a query."""
        # Initialize vector store for querying
        vector_store = Pinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        # Get relevant documents
        docs = vector_store.similarity_search(query, k=k)
        return docs