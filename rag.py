import os
import tempfile
import streamlit as st
import chromadb
import ollama

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are a grounded AI assistant.
Answer ONLY using the provided context.
If the context does not contain enough information, say so clearly.
Be precise, structured, and factual.
"""

# =========================
# VECTOR STORE
# =========================

def get_vector_collection():
    embedding_function = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )

    client = chromadb.PersistentClient(path="./chroma_store")

    return client.get_or_create_collection(
        name="rag_collection",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

# =========================
# DOCUMENT PROCESSING
# =========================

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp.write(uploaded_file.read())
    temp.close()

    loader = PyMuPDFLoader(temp.name)
    docs = loader.load()
    os.unlink(temp.name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    return splitter.split_documents(docs)


def add_documents_to_store(chunks: list[Document], file_name: str):
    collection = get_vector_collection()

    documents, metadatas, ids = [], [], []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)
        ids.append(f"{file_name}_{i}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

# =========================
# CONVERSATION MEMORY
# =========================

def init_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_to_memory(question, answer):
    st.session_state.chat_history.append(
        {"question": question, "answer": answer}
    )
    st.session_state.chat_history = st.session_state.chat_history[-5:]


def format_history():
    history = ""
    for turn in st.session_state.chat_history:
        history += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
    return history

# =========================
# QUERY REWRITE
# =========================

def rewrite_query(question: str) -> str:
    history = format_history()

    prompt = f"""
Rewrite the question so it is standalone and explicit.

Conversation:
{history}

Question: {question}
Rewritten Question:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()

# =========================
# RETRIEVAL
# =========================

def retrieve_documents(query: str, threshold: float = 0.75):
    collection = get_vector_collection()
    results = collection.query(query_texts=[query], n_results=10)

    docs = results["documents"][0]
    distances = results["distances"][0]

    return [doc for doc, dist in zip(docs, distances) if dist < threshold]

# =========================
# RE-RANKING
# =========================

def rerank(query: str, documents: list[str]):
    if not documents:
        return ""

    encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder.rank(query, documents, top_k=3)

    context = ""
    for r in ranks:
        context += documents[r["corpus_id"]] + "\n"

    return context.strip()

# =========================
# GENERATION
# =========================

def generate_answer(context: str, question: str):
    history = format_history()

    stream = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Conversation:
{history}

Context:
{context}

Question:
{question}
""",
            },
        ],
    )

    answer = ""
    for chunk in stream:
        if not chunk["done"]:
            token = chunk["message"]["content"]
            answer += token
            yield token

    add_to_memory(question, answer)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config("Production RAG", layout="wide")
st.title(" Production-Grade Conversational RAG")

init_memory()

with st.sidebar:
    st.header(" Upload PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button("Process"):
        chunks = process_document(uploaded_file)
        add_documents_to_store(chunks, uploaded_file.name)
        st.success("Document indexed successfully!")

st.divider()

question = st.text_area("Ask a question")
ask = st.button("Ask")

if ask and question:
    rewritten = rewrite_query(question)
    retrieved = retrieve_documents(rewritten)

    if not retrieved:
        st.warning("Not enough relevant information found.")
    else:
        context = rerank(rewritten, retrieved)
        st.write_stream(generate_answer(context, question))

        with st.expander(" Debug Info"):
            st.write("Rewritten Query:", rewritten)
            st.write("Context:", context)
