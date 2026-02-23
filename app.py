
import os, tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from resume_rag import (
    process_resume_file,
    analyze_resume,
    qa_resume,
    interview_questions,
    improve_resume,
    generate_resume,
)

load_dotenv()
app = Flask(__name__)


# ── Index ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Upload: extract text and return it ───────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file   = request.files["file"]
    suffix = Path(file.filename).suffix.lower()

    if suffix not in (".pdf", ".txt"):
        return jsonify({"error": "Only .pdf and .txt files are supported"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = process_resume_file(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

    return jsonify({
        "resume_text": text,
        "file_name":   file.filename,
        "chars":       len(text),
    })


# ── 1. Analyze ────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data   = request.json
    resume = data.get("resume", "").strip()
    role   = data.get("role", "").strip()

    if not resume: return jsonify({"error": "No resume text provided."}), 400
    if not role:   return jsonify({"error": "Target role is required."}), 400

    result = analyze_resume(resume, role, jd=data.get("jd", ""))
    return jsonify(result)


# ── 2. Q&A ────────────────────────────────────────────────────────
@app.route("/api/qa", methods=["POST"])
def qa():
    data     = request.json
    resume   = data.get("resume", "").strip()
    question = data.get("question", "").strip()

    if not resume:   return jsonify({"error": "No resume text provided."}), 400
    if not question: return jsonify({"error": "No question provided."}), 400

    return jsonify({"answer": qa_resume(resume, question)})


# ── 3. Interview Questions ────────────────────────────────────────
@app.route("/api/questions", methods=["POST"])
def questions():
    data   = request.json
    resume = data.get("resume", "").strip()

    if not resume: return jsonify({"error": "No resume text provided."}), 400

    result = interview_questions(
        resume_text   = resume,
        question_type = data.get("type", "Technical"),
        level         = data.get("level", "Medium"),
        count         = int(data.get("count", 5)),
        role          = data.get("role", ""),
    )
    return jsonify({"questions": result})


# ── 4. Improve ────────────────────────────────────────────────────
@app.route("/api/improve", methods=["POST"])
def improve():
    data   = request.json
    resume = data.get("resume", "").strip()

    if not resume: return jsonify({"error": "No resume text provided."}), 400

    result = improve_resume(
        resume_text = resume,
        area        = data.get("area", ""),
        role        = data.get("role", ""),
    )
    return jsonify({"suggestions": result})


# ── 5. Generate ───────────────────────────────────────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    role = data.get("role", "").strip()

    if not role: return jsonify({"error": "Target role is required."}), 400

    result = generate_resume(
        resume_text = data.get("resume", ""),
        role        = role,
        jd          = data.get("jd", ""),
    )
    return jsonify({"resume": result})


if __name__ == "__main__":
    app.run(debug=True, port=5000)