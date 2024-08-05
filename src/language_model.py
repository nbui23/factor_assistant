from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

def create_language_model():
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        load_in_8bit=True if torch.cuda.is_available() else False 
    )
    
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15,
        device=device
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def create_prompt_template():
    template = """[INST] You are an AI assistant specializing in the Factor programming language. Use the following relevant information to answer the user's question about Factor:

    {context}

    User question: {question}

    Provide a clear, accurate answer using the given information and your general programming knowledge. If you're unsure, say so. [/INST]
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])