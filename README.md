# 📄 Production-Grade Conversational RAG (PDF Chat)

A **production-oriented Retrieval-Augmented Generation (RAG)** system built with  
**Streamlit, ChromaDB, Ollama, and LangChain**.

Upload PDFs, ask conversational questions, and receive **grounded answers strictly from your documents** — fully **offline and local**.

---

## 🚀 Features

- 📄 PDF ingestion & chunking
- 🔍 Vector search using ChromaDB
- 🧠 Local embeddings via Ollama (`nomic-embed-text`)
- 🔁 Query rewriting using conversation history
- 📊 Cross-encoder re-ranking
- 💬 Conversational memory (last 5 turns)
- 🧾 Strict context-grounded answers (no hallucinations)
- ⚡ Streaming responses
- 🏠 Fully offline / local (no cloud APIs)

---

## 🧱 Architecture Overview

PDF Upload
↓
PyMuPDF Loader
↓
Text Chunking
↓
Ollama Embeddings
↓
ChromaDB (Persistent Vector Store)
↓
Query Rewrite (LLM)
↓
Vector Retrieval
↓
Cross-Encoder Re-Ranking
↓
Context Injection
↓
Answer Generation (Streaming)


---

## 🛠 Tech Stack

| Layer | Technology |
|-----|-----------|
| UI | Streamlit |
| LLM | Ollama (`llama3.2:3b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector DB | ChromaDB |
| PDF Loader | PyMuPDF |
| Chunking | LangChain |
| Re-Ranking | SentenceTransformers |
| Language | Python |

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/saurabhbhandariii/production-rag-streamlit.git
cd production-rag-streamlit
2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
🧠 Ollama Setup (Required)
Install Ollama from:
👉 https://ollama.com

Start Ollama:

ollama serve
Pull required models:

ollama pull llama3.2:3b
ollama pull nomic-embed-text
▶️ Run the Application
streamlit run app.py
Open in browser:

http://localhost:8501
🧪 How the RAG Pipeline Works
1. Document Ingestion
PDFs loaded using PyMuPDF

Chunked using recursive text splitting

2. Embedding & Storage
Each chunk embedded locally using Ollama

Stored in persistent ChromaDB

3. Query Rewrite
Converts follow-up questions into standalone queries

Uses conversation history

4. Retrieval
Vector similarity search

Distance threshold filtering

5. Re-Ranking
Cross-encoder ranks top relevant chunks

Improves answer precision

6. Answer Generation
Injects:

System prompt

Conversation history

Retrieved context

Model answers only from provided context

🔐 Grounding Rules
The system prompt enforces:

❌ No hallucinations

❌ No external knowledge

✅ Answers only from retrieved documents

✅ Clear fallback when info is missing
