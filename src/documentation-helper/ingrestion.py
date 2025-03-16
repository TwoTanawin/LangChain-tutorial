from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, OptimizersConfigDiff
from langchain_community.document_loaders import ReadTheDocsLoader
import torch
import time
import os

# Load environment variables
load_dotenv()

# Initialize Qdrant client with gRPC
qdrant_client = QdrantClient(
    url="localhost:6334",  # Use gRPC URL
    prefer_grpc=True  # Enable gRPC communication
)

collection_name = "docs_collection"

# Set device for embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

def ingest_docs():
    """Loads, processes, and indexes documents into Qdrant using gRPC."""
    
    # Load documents
    docs_path = r"E:\\Computing\\LLm-Engineer\\LangChain-tutorial\\src\\documentation-helper\\langchain-docs\\api.python.langchain.com\\en\\latest"
    
    # Check if the path exists
    if not os.path.exists(docs_path):
        print(f"❌ Path does not exist: {docs_path}")
        return
        
    loader = ReadTheDocsLoader(
        docs_path,
        encoding="utf-8"
    )
    
    try:
        raw_documents = loader.load()
        print(f"✅ Loaded {len(raw_documents)} documents")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # Fix document metadata
    for doc in documents:
        new_url = doc.metadata.get("source", "")
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    
    print(f"🔄 Processing {len(documents)} documents for indexing...")

    # Initialize embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )

    # Generate embeddings with error handling
    print("⏳ Generating embeddings...")
    embeddings = []
    for i, doc in enumerate(documents):
        try:
            if i % 100 == 0:
                print(f"Processed {i}/{len(documents)} embeddings")
            embeddings.append(embedding.embed_query(doc.page_content))
        except Exception as e:
            print(f"❌ Error embedding document {i}: {e}")
            continue

    if not embeddings:
        print("⚠ No embeddings generated. Exiting...")
        return

    vector_size = len(embeddings[0])
    
    # Check if collection exists and create if needed
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            print("🚀 Creating new Qdrant collection via gRPC with proper indexing...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance="Cosine"
                )
            )
            print("✅ Collection created successfully with indexing configuration.")
        else:
            print(f"✅ Collection '{collection_name}' exists.")
    except Exception as e:
        print(f"❌ Error checking or creating collection: {e}")
        return

    print("📤 Uploading data to Qdrant in batches...")

    batch_size = 50  
    total_docs = len(embeddings)

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch_embeddings = embeddings[i:end_idx]
        batch_texts = documents[i:end_idx]

        points = [
            PointStruct(
                id=i + idx,  
                vector=batch_embeddings[idx],
                payload={
                    "text": batch_texts[idx].page_content,
                    "source": batch_texts[idx].metadata.get("source", "")
                }
            )
            for idx in range(len(batch_embeddings))
        ]
        
        try:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            print(f"✅ Uploaded batch {i // batch_size + 1}/{(total_docs // batch_size) + 1}")
        except Exception as e:
            print(f"❌ Error uploading batch {i // batch_size + 1}: {e}")
    
    # Force index optimization
    print("🔄 Forcing index optimization...")
    try:
        qdrant_client.update_collection(
            collection_name=collection_name,
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=0,
                memmap_threshold=20000,
            )
        )
        print("✅ Optimization triggered.")
    except Exception as e:
        print(f"❌ Error optimizing collection: {e}")
    
    print("🎉 Data processing complete!")

if __name__ == "__main__":
    ingest_docs()