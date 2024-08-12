from src.config import CORPUS_PATH
from src.data_loader import load_corpus, prepare_documents, split_documents
from src.embeddings import create_vectorstore
from src.retriever import create_retriever
from src.language_model import create_language_model, create_prompt_template

def setup_qa_system():
    print("Setting up QA system...")
    corpus = load_corpus(CORPUS_PATH)
    documents = prepare_documents(corpus)
    splits = split_documents(documents)
    
    vectorstore = create_vectorstore(splits)
    retriever = create_retriever(vectorstore, k=5)
    
    llm = create_language_model()
    prompt_template = create_prompt_template()
    
    print("QA system setup complete.")
    return llm, retriever, prompt_template

def ask_question(llm, retriever, prompt_template, question):
    try:
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = prompt_template.format(context=context, question=question)
        
        print("Sending request to language model...")
        answer = llm(prompt)
        print("Received response from language model.")
        
        if len(answer.split()) < 10:
            elaboration_prompt = f"Please elaborate on this answer: {answer}"
            answer += "\n\n" + llm(elaboration_prompt)
        
        return answer
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        return f"An error occurred: {str(e)}"