import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_corpus(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_documents(corpus):
    return [f"Topic: {item['topic']}\n\nExplanation: {item['explanation']}\n\nExample: {item['example']}" for item in corpus]

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents(documents)