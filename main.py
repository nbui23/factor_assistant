from src.config import CORPUS_PATH
from src.data_loader import load_corpus, prepare_documents, split_documents
from src.embeddings import create_vectorstore
from src.retriever import create_retriever
from src.language_model import create_language_model, create_prompt_template
from src.qa_chain import create_qa_chain

def setup_qa_system():
    corpus = load_corpus(CORPUS_PATH)
    documents = prepare_documents(corpus)
    splits = split_documents(documents)
    
    vectorstore = create_vectorstore(splits)
    retriever = create_retriever(vectorstore)
    
    llm = create_language_model()
    prompt = create_prompt_template()
    
    return create_qa_chain(llm, retriever, prompt)

def ask_question(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"]

if __name__ == "__main__":
    qa_system = setup_qa_system()
    
    while True:
        question = input("Ask a question about Factor (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_question(qa_system, question)
        print(f"Answer: {answer}\n")