from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import torch

# Load environment variables
load_dotenv()

# Set device for HuggingFace model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Connect to Qdrant
print("Connecting to Qdrant...")
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "docs_collection"
collections = qdrant_client.get_collections()
print(collections)
collection_info = qdrant_client.get_collection(collection_name="docs_collection")
print(collection_info)

# Ensure Qdrant collection exists
if not qdrant_client.collection_exists(collection_name):
    print("Collection does not exist. Creating a new collection...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,  # Default size for all-mpnet-base-v2
            distance="Cosine"
        )
    )
    print("Collection created.")

def run_llm(query: str):
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )
    
    # Create vector store with Qdrant
    docsearch = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Initialize Chat Model
    chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768") 
    
    # Load retrieval QA prompt from LangChain hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Create document combination chain
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    # Create retrieval chain
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), 
        combine_docs_chain=stuff_documents_chain
    )
    
    # Invoke QA chain
    result = qa.invoke(input={'input': query})
    
    return result

if __name__ == "__main__":
    print("Hello ...")
    res = run_llm(query="What is LangChain?")
    
    # Print output safely
    if "answer" in res:
        print(res["answer"])
    else:
        print("Full Response:", res)
