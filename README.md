# Agentic RAG Assistant

This project is a retrieval-augmented generation (RAG) AI assistant designed to answer questions strictly based on a provided set of documents. It uses LangChain for orchestration and Groq's LLM for generation.

## Overview

The assistant allows users to query a knowledge base containing information about Artificial Intelligence, Biotechnology, Climate Science, Quantum Computing, Space Exploration, and Sustainable Energy. It is built to be an "Agentic" system that decides when to retrieve information and explicitly refuses to answer if the information is not present in its knowledge base.

## Architecture

1.  **Document Loading**: Text documents are loaded and split into chunks.
2.  **Embeddings & Vector Store**: Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS vector database.
3.  **Retrieval**: A retrieval mechanism fetches relevant chunks based on semantic similarity. A strict similarity threshold is enforced to filter out irrelevant information.
4.  **Agent/Chain**: A LangChain pipeline processes the user query and retrieved context.
5.  **LLM**: Groq (`llama-3.1-8b-instant`) generates the response based **only** on the context.

## Anti-Hallucination Measures

-   **Retrieval-Only Answering**: The system prompt explicitly forbids using external knowledge.
-   **Similarity Threshold**: If the retrieved documents do not match the query sufficiently (based on a score threshold), the system does not provide them to the context, leading to an "I don't know" response.
-   **Strict System Prompt**: The LLM is instructed to say "I do not have enough information" if the context is insufficient.

## How to Run Locally

### Prerequisites

-   Python 3.9+
-   A Groq API Key

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY=your_actual_api_key_here
    ```

### Running the App

1.  **Initialize the Knowledge Base**:
    The first time you run the app, it will create the vector store from the documents in `data/`.

2.  **Start the UI**:
    ```bash
    streamlit run ui/app.py
    ```
    Or use the helper script:
    ```bash
    python main.py
    ```

## Example Queries

-   **In-Scope**: "What is strong AI?", "How does quantum computing work?", "Explain the types of sustainable energy."
-   **Out-of-Scope**: "Who is the president of the USA?", "What is the capital of Australia?" (will be refused).

## Limitations

-   The knowledge is limited strictly to the provided text files.
-   It assumes the provided documents are the sole source of truth.

## Future Improvements

-   Add support for PDF and other file formats.
-   Implement conversation history memory (Multi-turn RAG).
-   Add citation sources to the output.
