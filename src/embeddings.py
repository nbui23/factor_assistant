from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.config import EMBEDDING_MODEL

def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(documents, embeddings)