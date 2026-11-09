import os
import time
from dotenv import load_dotenv
import chromadb
from google import genai
from pypdf import PdfReader

# 1. Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing from .env")

# 2. Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="documents")

# 3. Initialize Gemini Client
client = genai.Client(api_key=api_key)

def get_document_stats(pdf_path: str = "data/sample.pdf"):
    if not os.path.exists(pdf_path):
        return "Document not found."
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    full_text = " ".join([page.extract_text() or "" for page in reader.pages])
    words = len(full_text.split())
    chars = len(full_text)
    total_chunks = collection.count()
    return (
        f"📊 Document Statistics:\n"
        f"- Total Pages: {total_pages}\n"
        f"- Total Words: {words:,}\n"
        f"- Total Characters: {chars:,}\n"
        f"- Indexed Vector Chunks: {total_chunks}"
    )

def ask_document(query: str, n_results: int = 3):
    # Quick metadata intercept for document-level counts
    query_lower = query.lower()
    if any(k in query_lower for k in ["count", "total words", "number of words", "how many words", "how many pages"]):
        return get_document_stats()

    # Vector search for context
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    retrieved_chunks = results["documents"][0]
    context = "\n---\n".join(retrieved_chunks)

    prompt = f"""
You are an expert document assistant. Answer the user's question accurately using ONLY the context provided below. If the answer is not contained within the context, state clearly: "I cannot find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            err_text = str(e)
            if any(x in err_text for x in ["503", "UNAVAILABLE", "RemoteProtocolError", "disconnected"]):
                time.sleep(2)
                continue
            raise e

    return "Service temporarily busy. Please retry in a moment."

if __name__ == "__main__":
    print("=== Document Query Agent Ready (Type 'exit' to quit) ===\n")
    while True:
        user_query = input("Ask a question: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
        if not user_query.strip():
            continue
            
        print("\nSearching document & generating answer...")
        answer = ask_document(user_query)
        print(f"\nResponse:\n{answer}\n")
