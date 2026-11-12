try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import time
import streamlit as st
import chromadb
from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="DocQuery RAG Agent", page_icon="📄", layout="wide")

# Load environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY missing from .env file.")
    st.stop()

# Initialize ChromaDB and Gemini Client
@st.cache_resource
def get_clients():
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_or_create_collection(name="documents")
    gemini = genai.Client(api_key=api_key)
    return chroma, collection, gemini

chroma_client, collection, client = get_clients()

# Helper: Chunking
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Helper: Process Uploaded PDF
def process_uploaded_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            extracted_text += txt + "\n"

    chunks = chunk_text(extracted_text)
    if not chunks:
        return 0, 0, 0

    # Reset collection for the newly uploaded document
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Index new chunks
    ids = [f"chunk_{idx}" for idx in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    word_count = len(extracted_text.split())
    char_count = len(extracted_text)
    return len(reader.pages), word_count, len(chunks)

# Query logic
def query_agent(prompt_text: str, n_results: int = 3) -> str:
    results = collection.query(query_texts=[prompt_text], n_results=n_results)
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    
    if not retrieved_chunks:
        return "No indexed documents found. Please upload or ingest a PDF first."

    context = "\n---\n".join(retrieved_chunks)

    prompt = f"""
You are an expert document assistant. Answer the user's question accurately using ONLY the context provided below. If the answer is not contained within the context, state clearly: "I cannot find the answer in the provided document."

Context:
{context}

Question:
{prompt_text}

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
            return f"Error communicating with Gemini: {err_text}"
            
    return "Service temporarily busy. Please retry in a moment."

# --- UI Layout ---

with st.sidebar:
    st.title("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document", type="primary"):
            with st.spinner("Extracting text, chunking, and updating ChromaDB..."):
                pages, words, chunks = process_uploaded_pdf(uploaded_file)
                st.success(f"Ingested successfully!")
                st.metric("Total Pages", pages)
                st.metric("Total Words", f"{words:,}")
                st.metric("Vector Chunks", chunks)

    st.markdown("---")
    st.caption("Powered by ChromaDB local vector storage & Gemini 3.5 Flash")

st.title("Intelligent Document Query Agent")
st.write("Ask natural-language questions about your ingested document.")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query input
if user_prompt := st.chat_input("Ask a question about the document..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate grounded response
    with st.chat_message("assistant"):
        with st.spinner("Searching vector index & synthesizing response..."):
            answer = query_agent(user_prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})