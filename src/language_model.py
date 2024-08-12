from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def create_language_model():
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    print(f"API Token: {'*' * len(huggingfacehub_api_token) if huggingfacehub_api_token else 'Not found'}")
    
    if not huggingfacehub_api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.3, "max_length": 512},
            huggingfacehub_api_token=huggingfacehub_api_token
        )
        print("Language model created successfully")
        return llm
    except Exception as e:
        print(f"Error creating language model: {str(e)}")
        raise

def create_prompt_template():
    template = """You are an AI assistant specializing in the Factor programming language. Your task is to provide clear and accurate answers based on the given context and your knowledge of programming.

    Context information about Factor:
    {context}

    Human: {question}

    Assistant: Based on the context provided and my knowledge of programming languages, I can answer your question about Factor. 
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])