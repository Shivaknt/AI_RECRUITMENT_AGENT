
import os, re, hashlib, json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "resumes"
GEMINI_MODEL    = "gemini-2.5-flash"
ST_MODEL_NAME   = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100
TOP_K           = 6

# ── Gemini client ────────────────────────────────────────────────
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── ChromaDB (persistent) ────────────────────────────────────────
_chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
_collection = _chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ── Sentence Transformer — LAZY LOADED ──────────────────────────
# Not loaded at import time so Flask starts instantly.
# Loaded on first embed call, then reused forever.
_st_model = None

def _get_model():
    global _st_model
    if _st_model is None:
        print("⚙️  Loading embedding model (first time only)...")
        _st_model = SentenceTransformer(ST_MODEL_NAME)
        print("✅ Embedding model ready.")
    return _st_model


# ════════════════════════════════════════════════════════════════
# STEP 1 — EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_text(file_path: str) -> str:
    path   = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        pages = []
        with open(file_path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                t = page.extract_text()
                if t: pages.append(t.strip())
        return "\n\n".join(pages)
    else:
        raise ValueError(f"Unsupported type: {suffix}. Use .pdf or .txt")


# ════════════════════════════════════════════════════════════════
# STEP 2 — CHUNKING
# ════════════════════════════════════════════════════════════════

def chunk_text(text: str) -> list[str]:
    text   = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ════════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDINGS
# ════════════════════════════════════════════════════════════════

def embed_documents(chunks: list[str]) -> list[list[float]]:
    model = _get_model()
    vecs  = model.encode(chunks, batch_size=16, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()

def embed_query(query: str) -> list[float]:
    model = _get_model()
    vec   = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()


# ════════════════════════════════════════════════════════════════
# STEP 4 — VECTOR STORE
# ════════════════════════════════════════════════════════════════

def _resume_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def is_resume_stored(resume_text: str) -> bool:
    rid     = _resume_hash(resume_text)
    results = _collection.get(where={"resume_id": rid}, limit=1)
    return len(results["ids"]) > 0

def store_resume(resume_text: str, file_name: str = "resume") -> str:
    """Chunk → embed → store. Skips if already stored. Returns resume_id."""
    resume_id = _resume_hash(resume_text)
    if is_resume_stored(resume_text):
        print(f"✅ Already in ChromaDB ({resume_id[:8]}…) — skipping embed.")
        return resume_id

    print(f"📄 Chunking '{file_name}'...")
    chunks = chunk_text(resume_text)
    print(f"   → {len(chunks)} chunks | embedding now...")

    embeddings = embed_documents(chunks)
    ids        = [f"{resume_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas  = [{"resume_id": resume_id, "chunk_index": i, "file_name": file_name}
                  for i in range(len(chunks))]

    _collection.add(ids=ids, embeddings=embeddings,
                    documents=chunks, metadatas=metadatas)
    print(f"✅ Stored {len(chunks)} chunks in '{CHROMA_DB_PATH}'")
    return resume_id

def retrieve_chunks(query: str, resume_id: str,
                    resume_text: str = "", top_k: int = TOP_K) -> list[str]:
    """
    Retrieve top-k relevant chunks.
    Falls back to full resume text split if ChromaDB returns nothing
    (handles edge case where resume has very few chunks).
    """
    try:
        query_vec = embed_query(query)
        results   = _collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"resume_id": resume_id},
        )
        docs = results["documents"][0] if results["documents"] else []
        if docs:
            return docs
    except Exception as e:
        print(f"⚠️  ChromaDB retrieve error: {e}")

    # Fallback — slice the raw resume text directly
    print("↩️  Using raw resume text fallback.")
    if resume_text:
        chunks = chunk_text(resume_text)
        return chunks[:top_k]
    return []

def delete_resume(resume_text: str) -> None:
    resume_id = _resume_hash(resume_text)
    stored    = _collection.get(where={"resume_id": resume_id})
    if stored["ids"]:
        _collection.delete(ids=stored["ids"])
        print(f"🗑️  Deleted {len(stored['ids'])} chunks.")

def list_stored_resumes() -> list[dict]:
    all_items = _collection.get()
    seen: dict[str, str] = {}
    for meta in all_items.get("metadatas", []):
        rid = meta.get("resume_id")
        if rid and rid not in seen:
            seen[rid] = meta.get("file_name", "unknown")
    return [{"resume_id": k, "file_name": v} for k, v in seen.items()]


# ════════════════════════════════════════════════════════════════
# STEP 5 — GEMINI
# ════════════════════════════════════════════════════════════════

def _ctx(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(chunks)

def _call_gemini(prompt: str, max_tokens: int = 5000) -> str:
    response = _gemini_client.models.generate_content(
        model    = GEMINI_MODEL,
        contents = prompt,
        config   = types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return response.text.strip()


# ════════════════════════════════════════════════════════════════
# RAG FUNCTIONS
# ════════════════════════════════════════════════════════════════

def rag_analyze_resume(resume_text: str, role: str,
                       jd: str = "", file_name: str = "resume") -> dict:
    resume_id = store_resume(resume_text, file_name)
    query     = f"skills experience qualifications education achievements for {role}"
    chunks    = retrieve_chunks(query, resume_id, resume_text)
    context   = _ctx(chunks)

    jd_part = f"\n\nJob Description:\n{jd}" if jd else ""
    prompt  = f"""You are a senior HR recruiter. Analyze this resume for the role: {role}{jd_part}

RESUME:
{context}

Return ONLY valid JSON (no markdown, no extra text):
{{"score": <0-100>, "strengths": ["..."], "gaps": ["..."], "overall_summary": "..."}}"""

    raw = _call_gemini(prompt)
    raw = re.sub(r"^```[a-z]*\n?|```$", "", raw.strip()).strip()
    try:
        return json.loads(raw)
    except Exception:
        # try to extract JSON from anywhere in the response
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
        return {"score": 0, "strengths": [], "gaps": [],
                "overall_summary": raw, "parse_error": True}


def rag_qa(resume_text: str, question: str, file_name: str = "resume") -> str:
    resume_id = store_resume(resume_text, file_name)
    chunks    = retrieve_chunks(question, resume_id, resume_text)
    context   = _ctx(chunks)

    prompt = f"""You are a helpful assistant. Answer the question using the resume below.
Be specific and detailed. If truly not present, say so briefly.

RESUME:
{context}

QUESTION: {question}

Answer:"""
    return _call_gemini(prompt)


def rag_interview_questions(resume_text: str, question_type: str = "Technical",
                            level: str = "Medium", count: int = 5,
                            role: str = "", file_name: str = "resume") -> str:
    resume_id = store_resume(resume_text, file_name)
    query     = f"{question_type} skills experience {role}"
    chunks    = retrieve_chunks(query, resume_id, resume_text)
    context   = _ctx(chunks)

    role_part = f" for role: {role}" if role else ""
    prompt    = f"""Generate exactly {count} {level.lower()} {question_type.lower()} interview questions{role_part}.
Base them specifically on THIS candidate's experience. Number each 1., 2., etc.

RESUME:
{context}

Questions:"""
    return _call_gemini(prompt)


def rag_improve_resume(resume_text: str, area: str,
                       role: str = "", file_name: str = "resume") -> str:
    resume_id = store_resume(resume_text, file_name)
    chunks    = retrieve_chunks(f"{area} {role}", resume_id, resume_text)
    context   = _ctx(chunks)

    prompt = f"""You are a professional resume coach.
Area: {area}{f' | Target role: {role}' if role else ''}

Give specific, actionable improvements referencing actual content below.
Include before/after rewrites where useful.

RESUME:
{context}

Suggestions:"""
    return _call_gemini(prompt)


def rag_generate_resume(resume_text: str = "", role: str = "",
                        jd: str = "", file_name: str = "resume") -> str:
    context = ""
    if resume_text:
        resume_id = store_resume(resume_text, file_name)
        chunks    = retrieve_chunks(f"experience skills education {role}",
                                    resume_id, resume_text, top_k=8)
        context   = _ctx(chunks)

    prompt = f"""You are an expert resume writer. Generate a professional ATS-optimized resume.
Target Role: {role}
{f'Job Description:{chr(10)}{jd}' if jd else ''}
{f'Base on this experience:{chr(10)}{context}' if context else ''}

Format: Summary | Experience (with metrics) | Skills | Education. Plain text only."""
    return _call_gemini(prompt, max_tokens=2000)


def process_resume_file(file_path: str) -> tuple[str, str]:
    file_name = Path(file_path).name
    print(f"\n{'═'*50}\n Processing: {file_name}\n{'═'*50}")
    text      = extract_text(file_path)
    resume_id = store_resume(text, file_name)
    print(f"✅ Ready — resume_id: {resume_id[:8]}…\n")
    return text, resume_id


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Stored resumes:")
        for r in list_stored_resumes():
            print(f"  • {r['file_name']} [{r['resume_id'][:8]}…]")
    else:
        text, rid = process_resume_file(sys.argv[1])
        role      = sys.argv[2] if len(sys.argv) > 2 else "Software Engineer"
        result    = rag_analyze_resume(text, role=role)
        print(f"Score: {result.get('score')}/100")
        print(f"Summary: {result.get('overall_summary','')[:300]}")