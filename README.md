# Factor Programming Assistant

## Overview

The Factor Programming Assistant is a Retrieval-Augmented Generation (RAG) system designed to answer questions about the Factor programming language.

## Features

- **Comprehensive Knowledge Base**: Includes detailed information about Factor's syntax, concepts, best practices, and comparisons with other languages.
- **Interactive Q&A**: Users can ask questions about Factor and receive detailed, context-aware answers.
- **Retrieval-Augmented Generation**: Combines information retrieval with language generation for accurate and relevant responses.
- **Extensible Corpus**: The knowledge base can be easily updated or expanded to include new information about Factor.

## Technical Stack

- **Language Model**: Google's FLAN-T5 model
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **Framework**: LangChain for building the RAG pipeline

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/nbui23/factor_assistant.git
   cd factor_assistant
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Hugging Face API token:
   - Create a `.env` file in the project root
   - Add your Hugging Face API token: `HUGGINGFACEHUB_API_TOKEN=your_token_here`

## Usage

Run the main script to start the Factor Programming Assistant:

```
python main.py
```

You can then interact with the assistant by asking questions about the Factor programming language.

## Project Structure

- `main.py`: The entry point of the application
- `src/`:
  - `config.py`: Configuration settings
  - `data_loader.py`: Functions for loading and preprocessing the corpus
  - `embeddings.py`: Handles creation of embeddings and vector store
  - `language_model.py`: Defines the language model and prompt template
  - `retriever.py`: Sets up the retrieval system
  - `__init__.py`: Initializes the QA system
- `data/factor_corpus.json`: The comprehensive Factor knowledge base

## Extending the Corpus

To add new information to the Factor knowledge base:

1. Open `data/factor_corpus.json`
2. Add new entries following the existing format:
   ```json
   {
     "topic": "Your New Topic",
     "explanation": "Detailed explanation here",
     "example": "Code example or usage"
   }
   ```
3. Save the file and restart the application

