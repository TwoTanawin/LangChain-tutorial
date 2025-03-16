from qdrant_client import QdrantClient
from qdrant_client.models import OptimizersConfigDiff
import time

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="localhost:6334",
    prefer_grpc=True
)

collection_name = "docs_collection"

# First, let's get more detailed information about the collection
print("📊 Getting detailed collection info...")
try:
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    print(f"Collection info: {collection_info}")
    
    # Print specific details about vectors config
    print(f"Vector size: {collection_info.config.params.vectors.size}")
    print(f"Distance: {collection_info.config.params.vectors.distance}")
    
    # Check current optimizer configs
    print(f"Current optimizer configs: {collection_info.config.optimizer_config}")
except Exception as e:
    print(f"❌ Error getting collection details: {e}")

# Try forcing a collection optimization
print("\n🔄 Forcing collection optimization...")
try:
    # Option 1: Update collection with explicit optimization config
    qdrant_client.update_collection(
        collection_name=collection_name,
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,  # Force immediate indexing
            memmap_threshold=0,    # Use memory mapping for all vectors
        )
    )
    
    # Option 2: Directly call optimize method (available in newer Qdrant versions)
    print("⚙️ Calling collection optimize method...")
    try:
        qdrant_client.optimize_collection(
            collection_name=collection_name,
        )
        print("✅ Collection optimization triggered.")
    except Exception as e:
        print(f"⚠️ Direct optimization call failed (this is okay if using older Qdrant version): {e}")
    
    # Wait for indexing to complete
    print("⏳ Waiting for indexing to complete...")
    for i in range(5):  # Check multiple times with increasing waits
        time.sleep(5 * (i + 1))  # Progressive waiting
        try:
            info = qdrant_client.get_collection(collection_name=collection_name)
            print(f"Check {i+1}: Points count: {info.points_count}")
            # Safely check vectors_count
            if hasattr(info, 'vectors_count') and info.vectors_count is not None:
                print(f"Check {i+1}: Indexed vectors: {info.vectors_count}")
            else:
                print(f"Check {i+1}: Indexed vectors attribute not available")
                
            # Alternative method to check index status
            print("Checking collection clustering info...")
            try:
                cluster_info = qdrant_client.collection_cluster_info(collection_name=collection_name)
                print(f"Cluster info: {cluster_info}")
            except Exception as e:
                print(f"❌ Error getting cluster info: {e}")
        except Exception as e:
            print(f"❌ Error checking collection status: {e}")
            
    # Last attempt: recreate the index by recreating collection (this is risky!)
    print("\n⚠️ If indexing is still not working, consider recreating the collection.")
    print("You can do this by:")
    print("1. Exporting your data first")
    print("2. Deleting the collection")
    print("3. Creating a new collection with proper indexing parameters")
    print("4. Re-importing your data")
    
except Exception as e:
    print(f"❌ Error during optimization: {e}")

print("\n📋 Final recommendations:")
print("1. Check your Qdrant version - you might need to upgrade to get better indexing support")
print("2. Verify the Qdrant server is running with enough resources")
print("3. Consider consulting Qdrant documentation for your specific version")
print("4. If problems persist, you may need to recreate the collection with proper settings")