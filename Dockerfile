FROM python:3.10-slim

# -------- Environment --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------- System Dependencies --------
RUN apt-get update && apt-get install -y \
    build-essential \
        curl \
	    && rm -rf /var/lib/apt/lists/*

	    # -------- Working Directory --------
	    WORKDIR /app

	    # -------- Copy Requirements First (for layer caching) --------
	    COPY requirements.txt .

	    # -------- Install Python Dependencies --------
	    RUN pip install --upgrade pip
	    RUN pip install --no-cache-dir -r requirements.txt

	    # -------- Copy Project Files --------
	    COPY . .

	    # -------- Create Chroma Persistence Directory --------
	    RUN mkdir -p /app/chroma_db

	    # -------- Expose Flask Port --------
	    EXPOSE 5000

	    # -------- Run Application --------
	    CMD ["python", "app.py"]
