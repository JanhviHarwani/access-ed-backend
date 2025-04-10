from pinecone import Pinecone
import os

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Connect to the index
index = pc.Index("accessibility-index")  # Replace with your actual index name

# Fetch index statistics, including namespaces
index_stats = index.describe_index_stats()

# Print the namespaces
print("Available namespaces:", index_stats.get("namespaces", {}))
