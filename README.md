# Intelligent Document Query Agent (RAG System)

A Retrieval-Augmented Generation (RAG) system built with Python, ChromaDB, and Google Gemini API to query and extract insights from complex PDF documents without hallucinations.

## Architecture & Pipeline
1. **Document Ingestion (`ingest.py`)**: 
   - Extracts raw text from multi-page PDFs using `pypdf`.
   - Splits document text into overlapping chunks (400 words, 40-word overlap) to preserve context across boundaries.
   - Computes dense vector embeddings locally using ChromaDB's ONNX embedding engine (`all-MiniLM-L6-v2`).
   - Persists embeddings and chunk text inside a local Chroma vector database (`./chroma_db`).

2. **Retrieval & Grounded Generation (`query.py`)**:
   - Queries the ChromaDB index using cosine similarity to retrieve the top $k$ relevant context chunks.
   - Evaluates document metadata queries (word/character/page counts) directly from source files.
   - Formulates a grounded prompt restricting the LLM to verified context.
   - Generates answers using `gemini-3.5-flash` with built-in retry and connection-drop fault tolerance.

## Tech Stack
- **Language:** Python 3.12
- **Vector Database:** ChromaDB
- **LLM Engine:** Google Gemini API (`google-genai` SDK)
- **Document Processing:** `pypdf`
- **Environment Management:** `python-dotenv`

## Setup & Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<muhammadxel>/doc-query-rag.git
   cd doc-query-rag
