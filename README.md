# AI Recruitment Agent 🤖
### RAG Pipeline · Gemini LLM · Sentence Transformers · ChromaDB

---

## Stack

| Layer | Tool | Cost |
|---|---|---|
| LLM | Gemini 2.5 Flash | Free tier (generous) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` | **100% Free** (runs locally) |
| Vector DB | ChromaDB | **100% Free** (local folder) |
| PDF Extract | PyPDF2 | **100% Free** |


---

## Architecture

```
PDF / TXT Upload
    │
    ▼
extract_text()              PyPDF2 or plain read
    │
    ▼
chunk_text()                400-char chunks, 80-char overlap
    │
    ▼
MD5 hash check ─────────── Already in ChromaDB? → SKIP (free)
    │  (cache miss)
    ▼
embed_documents()           Sentence Transformers (local CPU/GPU)
    │                       model: all-MiniLM-L6-v2, 384-dim
    ▼
ChromaDB store              Saved to ./chroma_db/ (permanent)

─────────────────────────────────────────────────────

User Query (any section)
    │
    ▼
embed_query()               Same ST model (local, instant)
    │
    ▼
retrieve_chunks()           Top-5 cosine similar chunks
    │
    ▼
_call_gemini()              Gemini 1.5 Flash with context
    │
    ▼
Response to user
```

---

## Project Structure

```
recruitment-agent/
├── app.py                  Flask routes
├── resume_rag.py           Full RAG pipeline  ← main file
├── requirements.txt
├── .env                    Gemini API key only
├── chroma_db/              Auto-created vector store
└── templates/
    └── index.html          Frontend
```

---

## Setup

### 1. Get Gemini API key (free)
https://aistudio.google.com/app/apikey

### 2. Set .env
```
GEMINI_API_KEY=AIza...your_key_here
```

### 3. Install
```bash
pip install -r requirements.txt
```
> First run downloads the ST model (~80MB) once.
> Cached at `~/.cache/huggingface/` — never downloaded again.

### 4. Run
```bash
python app.py
# → http://localhost:5000
```

---

## Test the pipeline from terminal

```bash
# Analyze a resume
python resume_rag.py path/to/resume.pdf "Data Scientist"

# List all stored resumes
python resume_rag.py
```

---

## Key behaviour

- **Same resume uploaded twice** → ChromaDB already has it → zero embedding cost
- **Embedding model** downloads once, then runs fully offline forever
- **chroma_db/** persists between server restarts — no data loss
- **All 5 app sections** share the same stored embeddings — no duplication
