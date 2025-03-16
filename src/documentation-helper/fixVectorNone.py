# from qdrant_client import QdrantClient
# from qdrant_client.models import OptimizersConfigDiff

# # Connect to Qdrant
# qdrant_client = QdrantClient(url="localhost:6334", prefer_grpc=True)

# collection_name = "docs_collection"

# # Force index optimization
# qdrant_client.update_collection(
#     collection_name=collection_name,
#     optimizers_config=OptimizersConfigDiff(
#         default_segment_number=0,  # Forces optimization
#         memmap_threshold=20000,
#         indexing_threshold=1000,  # Ensure indexing occurs
#     )
# )

# print("🔄 Optimization triggered. Waiting for indexing to complete...")

# # Check if vectors are indexed
# import time
# max_wait_time = 120  # Wait up to 2 minutes
# start_time = time.time()

# while time.time() - start_time < max_wait_time:
#     collection_info = qdrant_client.get_collection(collection_name=collection_name)
#     vectors_count = getattr(collection_info, 'vectors_count', 0)
#     indexed_vectors_count = getattr(collection_info, 'indexed_vectors_count', 0)
#     points_count = getattr(collection_info, 'points_count', 0)

#     print(f"📊 Indexed vectors: {indexed_vectors_count}/{points_count}")
    
#     if indexed_vectors_count == points_count and indexed_vectors_count > 0:
#         print("✅ Indexing complete!")
#         break
#     else:
#         time.sleep(5)  # Wait for 5 seconds before checking again

# if indexed_vectors_count != points_count:
#     print("⚠ Indexing is still incomplete. Manual optimization may be needed.")
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

qdrant_client = QdrantClient(url="localhost:6334", prefer_grpc=True)

collection_name = "docs_collection"

# Check if the collection is searchable
search_result = qdrant_client.search(
    collection_name=collection_name,
    query_vector=[0.1] * 768,  # Dummy vector of the right size
    limit=5
)

print("Search Results:", search_result)
