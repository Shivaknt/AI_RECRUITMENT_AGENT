import os, re, json
from pathlib import Path

from google import genai
from google.genai import types
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

# ── Gemini client (single instance, reused for all calls) ─────────
_client = genai.Client(api_key=GEMINI_API_KEY)


# ════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_text(file_path: str) -> str:
    """Extract plain text from a PDF or TXT file."""
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore").strip()

    elif suffix == ".pdf":
        pages = []
        with open(file_path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                t = page.extract_text()
                if t:
                    pages.append(t.strip())
        return "\n\n".join(pages).strip()

    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf or .txt")


# ════════════════════════════════════════════════════════════════
# GEMINI CALL
# ════════════════════════════════════════════════════════════════

def _gemini(prompt: str, max_tokens: int = 5000) -> str:
    """Send prompt to Gemini, return response text."""
    response = _client.models.generate_content(
        model    = GEMINI_MODEL,
        contents = prompt,
        config   = types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return response.text.strip()


# ════════════════════════════════════════════════════════════════
# SECTION 1 — RESUME ANALYSIS
# ════════════════════════════════════════════════════════════════

def analyze_resume(resume_text: str, role: str, jd: str = "") -> dict:
    """Score the resume against a role. Returns {score, strengths, gaps, overall_summary}."""
    jd_part = f"\n\nJob Description:\n{jd}" if jd else ""

    prompt = f"""You are a senior HR recruiter. Analyze this resume for the role: {role}{jd_part}

RESUME:
{resume_text}

Return ONLY a valid JSON object — no markdown, no explanation, nothing else:
{{"score": <integer 0-100>, "strengths": ["...", "..."], "gaps": ["...", "..."], "overall_summary": "2-3 sentence assessment"}}

Scoring: 90-100 exceptional, 75-89 strong match, 50-74 partial, below 50 poor."""

    raw = _gemini(prompt)
    # Strip markdown fences if Gemini adds them
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip()).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON object from anywhere in the response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {
            "score": 0, "strengths": [], "gaps": [],
            "overall_summary": raw, "parse_error": True
        }


# ════════════════════════════════════════════════════════════════
# SECTION 2 — Q&A
# ════════════════════════════════════════════════════════════════

def qa_resume(resume_text: str, question: str) -> str:
    """Answer a question based on the full resume."""
    prompt = f"""You are a helpful assistant. Answer the question using the resume below.
Be specific and reference actual details from the resume.

RESUME:
{resume_text}

QUESTION: {question}

Answer:"""
    return _gemini(prompt)


# ════════════════════════════════════════════════════════════════
# SECTION 3 — INTERVIEW QUESTIONS
# ════════════════════════════════════════════════════════════════

def interview_questions(resume_text: str, question_type: str = "Technical",
                        level: str = "Medium", count: int = 5, role: str = "") -> str:
    """Generate tailored interview questions from the resume."""
    role_part = f" for the role of {role}" if role else ""
    level_guide = {
        "Easy":   "basic, tests fundamental understanding",
        "Medium": "requires explanation and depth",
        "Hard":   "complex, tests expert-level thinking and edge cases"
    }.get(level, "requires explanation and depth")

    prompt = f"""You are an expert interviewer. Generate exactly {count} {level.lower()} difficulty {question_type.lower()} interview questions{role_part}.

Rules:
- Questions must reference THIS candidate's actual experience, technologies, and companies
- Do NOT ask generic questions — make them specific to what's in the resume
- Number each: 1. 2. 3. ...
- {level} means: {level_guide}

RESUME:
{resume_text}

Interview Questions:"""
    return _gemini(prompt)


# ════════════════════════════════════════════════════════════════
# SECTION 4 — IMPROVEMENT TIPS
# ════════════════════════════════════════════════════════════════

def improve_resume(resume_text: str, area: str, role: str = "") -> str:
    """Give specific improvement suggestions for a resume area."""
    role_part = f"\nTarget Role: {role}" if role else ""

    prompt = f"""You are a professional resume coach with 15 years of experience.

Area to improve: {area}{role_part}

Study the resume below carefully and give SPECIFIC, ACTIONABLE suggestions.
- Quote actual lines from their resume and show how to rewrite them
- Give before → after examples
- Be direct and practical

RESUME:
{resume_text}

Improvement Suggestions:"""
    return _gemini(prompt)


# ════════════════════════════════════════════════════════════════
# SECTION 5 — GENERATE RESUME
# ════════════════════════════════════════════════════════════════

def generate_resume(resume_text: str = "", role: str = "", jd: str = "") -> str:
    """Generate a professional ATS-optimized resume."""
    base_part = f"\n\nBase it on this candidate's real experience:\n{resume_text}" if resume_text else ""
    jd_part   = f"\n\nTailor for this Job Description:\n{jd}" if jd else ""

    prompt = f"""You are an expert resume writer specialising in ATS-optimised resumes.

Generate a complete, polished resume for: {role}{jd_part}{base_part}

Format (plain text, no markdown):
- Header: Name | Email | Phone | LinkedIn | Location
- Summary: 3-4 compelling sentences tailored to the role
- Work Experience: Company | Title | Dates — bullet points with metrics and impact
- Skills: grouped by category
- Education: Degree | Institution | Year

Make it specific, results-driven, and ATS-friendly. Use strong action verbs and numbers."""
    return _gemini(prompt, max_tokens=2000)


# ════════════════════════════════════════════════════════════════
# FILE PROCESSING HELPER
# ════════════════════════════════════════════════════════════════

def process_resume_file(file_path: str) -> str:
    """Extract text from a PDF or TXT file. Returns the resume text."""
    file_name = Path(file_path).name
    print(f"📄 Extracting text from: {file_name}")
    text = extract_text(file_path)
    print(f"✅ Extracted {len(text)} characters")
    return text


# ════════════════════════════════════════════════════════════════
# CLI TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_rag.py path/to/resume.pdf [role]")
        sys.exit(0)

    text = process_resume_file(sys.argv[1])
    role = sys.argv[2] if len(sys.argv) > 2 else "Software Engineer"

    print(f"\n── Analysis for: {role}")
    result = analyze_resume(text, role=role)
    print(f"Score    : {result.get('score')}/100")
    print(f"Strengths: {result.get('strengths', [])[:2]}")
    print(f"Gaps     : {result.get('gaps', [])[:2]}")
    print(f"Summary  : {result.get('overall_summary', '')[:250]}")

    print(f"\n── Sample Q&A")
    print(qa_resume(text, "What are the top technical skills of this candidate?")[:300])