from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()


if __name__=="__main__":
    print("hi ...")
    
    pdf_path = r"E:\Computing\LLm-Engineer\LangChain-tutorial\src\vectors-in-memory\1706.03762v7.pdf"
    loader = PyMuPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n",
    )
    docs = text_splitter.split_documents(documents=documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorStore = FAISS.from_documents(docs, embeddings)
    vectorStore.save_local("faiss_index_react")
    
    new_vectorStore = FAISS.load_local(
        "faiss_index_react", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    llm = ChatGroq(model_name="mixtral-8x7b-32768")
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, 
        retrieval_qa_chat_prompt
    )
    
    retrival_chain = create_retrieval_chain(
        retriever=new_vectorStore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    
    
    
    result = retrival_chain.invoke(input={"input": 'Give me the gist of Transformer in 3 sentences'})

    print(result['answer'])
    