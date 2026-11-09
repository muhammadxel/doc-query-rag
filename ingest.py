import os
from pypdf import PdfReader
from dotenv import load_dotenv
import chromadb

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please check your .env configuration.")

# 2. Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# 3. Chunk text into overlapping segments
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# 4. Ingest and Embed into ChromaDB Vector Store
def ingest_document(pdf_path: str):
    print(f"Reading {pdf_path}...")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    print(f"Created {len(chunks)} text chunks.")

    # Persistent ChromaDB client (uses default local embeddings automatically)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="documents")

    print("Indexing text chunks into vector database...")
    ids = [f"chunk_{idx}" for idx in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        ids=ids
    )

    print(f"Ingestion complete! {len(chunks)} chunks stored in ./chroma_db.")

if __name__ == "__main__":
    pdf_file = "data/sample.pdf"
    if os.path.exists(pdf_file):
        ingest_document(pdf_file)
    else:
        print("Please place a PDF named 'sample.pdf' inside the 'data/' folder.")
