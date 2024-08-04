from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.config import LLM_MODEL
import torch

def create_language_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16, device_map="auto")
    
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def create_prompt_template():
    template = """[INST] You are an AI assistant specializing in the Factor programming language. Use the following relevant information to answer the user's question about Factor:

    {context}

    User question: {question}

    Provide a clear, accurate answer using the given information and your general programming knowledge. If you're unsure, say so. [/INST]
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])