FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le pyproject uniquement (cache Docker)
COPY pyproject.toml /app/

# Installer les deps "main" (sans installer le package)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Copier l'app
COPY app/ /app/app/

EXPOSE 7860
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]