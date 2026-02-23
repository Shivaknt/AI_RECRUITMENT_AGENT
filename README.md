# RecruitAI — Intelligent Hiring Suite

> AI-powered resume analysis, Q&A, interview prep, improvement coaching, and resume generation — built on **Gemini 1.5 Flash** with a clean Flask backend.

---

## What It Does

| Module | Description |
|--------|-------------|
| 🔍 **Resume Analysis** | Scores resume 0–100 against a role/JD. Extracts strengths and gaps. Cutoff: 75/100 |
| 💬 **Resume Q&A** | Ask anything about the resume — Gemini answers from the full text |
| 🎯 **Interview Prep** | Generates personalised questions by type (Technical/Behavioral/Managerial), difficulty, and count |
| ✨ **Improvement Tips** | Area-specific coaching with before/after rewrites |
| 🚀 **Generate Resume** | Produces a polished ATS-optimised resume tailored to the role and JD |

---

## Tech Stack

```
Frontend   →  Vanilla HTML + CSS + JS  (no framework)
Backend    →  Flask (Python)
LLM        →  Gemini 1.5 Flash via google-genai SDK
PDF Parse  →  PyPDF2
Container  →  Docker
```

No vector database. No embeddings. No ChromaDB.
A resume is ~1500 tokens — Gemini's 1M token context handles it directly.

---

## Project Structure

```
recruitment-agent/
├── app.py              ← Flask routes (5 API endpoints + upload)
├── resume_rag.py       ← Text extraction + Gemini calls
├── requirements.txt    ← 4 dependencies only
├── Dockerfile          ← Production container
├── .dockerignore       ← Docker build exclusions
├── .env                ← Your API key (never commit this)
└── templates/
    └── index.html      ← Full frontend (single file)
```

---

## Quick Start (Local)

### 1. Get a Gemini API Key

Free tier at → [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Generous free limits: 15 requests/min · 1M tokens/day

### 2. Clone and configure

```bash
git clone <your-repo>
cd recruitment-agent

# Create your .env file
echo "GEMINI_API_KEY=AIza...your_key_here" > .env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python app.py
```

Open → [http://localhost:5000](http://localhost:5000)

---

