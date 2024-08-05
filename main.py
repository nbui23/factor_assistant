from src.config import CORPUS_PATH
from src.data_loader import load_corpus, prepare_documents, split_documents
from src.embeddings import create_vectorstore
from src.retriever import create_retriever
from src.language_model import create_language_model, create_prompt_template
from langchain.chains import RetrievalQA

def setup_qa_system():
    corpus = load_corpus(CORPUS_PATH)
    documents = prepare_documents(corpus)
    splits = split_documents(documents)
    
    vectorstore = create_vectorstore(splits)
    retriever = create_retriever(vectorstore)
    
    llm = create_language_model()
    prompt = create_prompt_template()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

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