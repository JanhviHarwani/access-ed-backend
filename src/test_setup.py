# src/test_setup.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def test_connections():
    load_dotenv()
    
    print("Testing connections...")
    
    print("\nChecking environment variables:")
    required_vars = ['PINECONE_API_KEY']
    for var in required_vars:
        if os.getenv(var):
            print(f"✓ {var} found")
        else:
            print(f"✗ {var} missing")
    
    print("\nTesting Pinecone connection:")
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        indexes = pc.list_indexes()
        print("✓ Pinecone connection successful")
        print(f"✓ Available indexes: {indexes.names()}")
    except Exception as e:
        print(f"✗ Pinecone connection failed: {str(e)}")
    
    print("\nTesting HuggingFace embeddings:")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        test_text = "Testing embeddings"
        embed = embeddings.embed_query(test_text)
        print(f"✓ Embeddings working (vector size: {len(embed)})")
    except Exception as e:
        print(f"✗ Embeddings failed: {str(e)}")

if __name__ == "__main__":
    test_connections()