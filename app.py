
import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from resume_rag import (
    process_resume_file,
    list_stored_resumes,
    delete_resume,
    rag_analyze_resume,
    rag_qa,
    rag_interview_questions,
    rag_improve_resume,
    rag_generate_resume,
)

load_dotenv()
app = Flask(__name__)


# ── Index ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Upload & embed resume ─────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accept a file upload (PDF or TXT), extract text,
    embed & store in ChromaDB (skips if already stored).
    Returns resume text so frontend can cache it.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        return jsonify({"error": "Only .pdf and .txt files are supported"}), 400

    # Save temp file so PyPDF2 can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text, resume_id = process_resume_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return jsonify({
        "resume_text": text,
        "resume_id":   resume_id,
        "file_name":   file.filename,
        "message":     "Resume processed and embeddings stored."
    })


# ── 1. Analyze ───────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data      = request.json
    resume    = data.get("resume", "")
    role      = data.get("role", "")
    jd        = data.get("jd", "")
    file_name = data.get("file_name", "resume")

    if not resume or not role:
        return jsonify({"error": "Resume text and role are required."}), 400

    result = rag_analyze_resume(resume, role, jd, file_name)
    return jsonify(result)


# ── 2. Q&A ───────────────────────────────────────────────────────
@app.route("/api/qa", methods=["POST"])
def qa():
    data      = request.json
    resume    = data.get("resume", "")
    question  = data.get("question", "")
    file_name = data.get("file_name", "resume")

    if not resume:
        return jsonify({"error": "No resume text provided."}), 400
    if not question:
        return jsonify({"error": "No question provided."}), 400

    answer = rag_qa(resume, question, file_name)
    return jsonify({"answer": answer})


# ── 3. Interview Questions ────────────────────────────────────────
@app.route("/api/questions", methods=["POST"])
def questions():
    data      = request.json
    resume    = data.get("resume", "")
    file_name = data.get("file_name", "resume")

    if not resume:
        return jsonify({"error": "No resume text provided."}), 400

    result = rag_interview_questions(
        resume_text   = resume,
        question_type = data.get("type", "Technical"),
        level         = data.get("level", "Medium"),
        count         = int(data.get("count", 5)),
        role          = data.get("role", ""),
        file_name     = file_name
    )
    return jsonify({"questions": result})


# ── 4. Improve ───────────────────────────────────────────────────
@app.route("/api/improve", methods=["POST"])
def improve():
    data      = request.json
    resume    = data.get("resume", "")
    file_name = data.get("file_name", "resume")

    if not resume:
        return jsonify({"error": "No resume text provided."}), 400

    result = rag_improve_resume(
        resume_text = resume,
        area        = data.get("area", ""),
        role        = data.get("role", ""),
        file_name   = file_name
    )
    return jsonify({"suggestions": result})


# ── 5. Generate ──────────────────────────────────────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    data      = request.json
    resume    = data.get("resume", "")
    role      = data.get("role", "")
    file_name = data.get("file_name", "resume")

    if not role:
        return jsonify({"error": "Target role is required."}), 400

    result = rag_generate_resume(
        resume_text = resume,
        role        = role,
        jd          = data.get("jd", ""),
        file_name   = file_name
    )
    return jsonify({"resume": result})


# ── Utility: list stored resumes ─────────────────────────────────
@app.route("/api/stored-resumes", methods=["GET"])
def stored_resumes():
    return jsonify(list_stored_resumes())


# ── Utility: delete a stored resume ──────────────────────────────
@app.route("/api/delete-resume", methods=["POST"])
def delete_stored():
    resume = request.json.get("resume", "")
    if not resume:
        return jsonify({"error": "Resume text required"}), 400
    delete_resume(resume)
    return jsonify({"message": "Deleted."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
