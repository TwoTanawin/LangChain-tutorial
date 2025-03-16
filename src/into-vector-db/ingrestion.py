from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

if __name__ == "__main__":
    print("Ingestion....")
    loader = TextLoader(r"E:\Computing\LLm-Engineer\LangChain-tutorial\src\into-vector-db\mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    
    print("Splitting ...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    print("Generating embeddings...")
    embeddings = [embedding.embed_query(text.page_content) for text in texts]
    
    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(host="localhost", port=6333)  # Adjust host/port if needed
    
    collection_name = "my_collection"
    from qdrant_client.models import VectorParams

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=len(embeddings[0]),
            distance="Cosine"
        )
    )
    
    print("Pushing data to Qdrant...")
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": texts[i].page_content}
        )
        for i in range(len(embeddings))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    
    print("Data pushed to Qdrant successfully!")