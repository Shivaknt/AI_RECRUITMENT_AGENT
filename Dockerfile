# ── Stage: Python 3.11 slim base ─────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="RecruitAI"
LABEL description="AI Recruitment Agent — Gemini powered resume analysis suite"

# Set working directory
WORKDIR /app

# Install system dependencies needed by PyPDF2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY resume_rag.py .
COPY templates/ templates/
COPY static/ static/

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Flask port
EXPOSE 5000

# Environment defaults (override via docker run -e or .env)
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run with Gunicorn in production (falls back to Flask dev server if not installed)
CMD ["python", "app.py"]