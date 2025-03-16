from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

if __name__=="__main__":
    print("Retrieving ...")
    
    client = QdrantClient(":memory:")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatGroq(model_name="mixtral-8x7b-32768")
    
    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm | StrOutputParser()
    # result = chain.invoke(input={})
    # print(result.content)
    
    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(host="localhost", port=6333)
    
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="my_collection",
        embedding=embeddings,
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    
    result = retrival_chain.invoke(input={"input": query})

    print(result)