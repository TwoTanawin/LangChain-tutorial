from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.document_loaders import ReadTheDocsLoader
import torch
import time
import os

# Load environment variables
load_dotenv()

# Initialize Qdrant client with a much longer timeout
qdrant_client = QdrantClient(host="localhost", port=6333)

collection_name = "docs_collection"

# Set device for embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

def ingest_docs():
    """Loads, processes, and indexes documents into Qdrant."""
    
    # Load documents
    docs_path = r"E:\Computing\LLm-Engineer\LangChain-tutorial\src\documentation-helper\langchain-docs\api.python.langchain.com\en\latest"
    
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
            # Skip this document but continue with others
            continue

    if not embeddings:
        print("⚠ No embeddings generated. Exiting...")
        return

    vector_size = len(embeddings[0])
    
    # Check if collection exists and create if needed
    try:
        collection_exists = False
        collections = qdrant_client.get_collections().collections
        for collection in collections:
            if collection.name == collection_name:
                collection_exists = True
                print(f"✅ Collection '{collection_name}' exists.")
                break
                
        if not collection_exists:
            print("🚀 Creating new Qdrant collection...")
            # Create collection with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    print("✅ Collection created successfully.")
                    break
                except Exception as e:
                    print(f"❌ Attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)  # Exponential backoff
                        print(f"⏳ Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print("❌ Failed to create collection after multiple attempts.")
                        return
    except Exception as e:
        print(f"❌ Error checking collections: {e}")
        return

    print("📤 Uploading data to Qdrant in batches...")

    # Reduce batch size to avoid timeouts
    batch_size = 50  # Reduced from 1000 to 100
    total_docs = len(embeddings)

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch_embeddings = embeddings[i:end_idx]
        batch_texts = documents[i:end_idx]

        points = [
            PointStruct(
                id=(i * batch_size) + idx,  # Ensure globally unique IDs
                vector=batch_embeddings[idx],
                payload={
                    "text": batch_texts[idx].page_content,
                    "source": batch_texts[idx].metadata.get("source", "")
                }
            )
            for idx in range(len(batch_embeddings))
        ]
        
        # Upload batch with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                qdrant_client.upsert(collection_name=collection_name, points=points)
                print(f"✅ Uploaded batch {i // batch_size + 1}/{(total_docs // batch_size) + 1}")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt+1}/{max_retries} failed for batch {i // batch_size + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"⏳ Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed to upload batch {i // batch_size + 1} after multiple attempts.")

    print("🎉 Data processing complete!")

if __name__ == "__main__":
    ingest_docs()