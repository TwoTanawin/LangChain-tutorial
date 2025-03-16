from qdrant_client import QdrantClient

# Connect using gRPC
client = QdrantClient(url="localhost:6334", prefer_grpc=True)

# Get list of collections
collections = client.get_collections()
print("Available Collections:", collections)

# Check collection details (Replace 'docs_collection' with your collection name)
collection_name = "docs_collection"
collection_info = client.get_collection(collection_name=collection_name)
print("Collection Info:", collection_info)
