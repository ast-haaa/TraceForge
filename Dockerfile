FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_PROVIDER=stub \
    AGENTREPLAY_READ_ONLY=1 \
    DATABASE_URL=sqlite:////app/agentreplay.db \
    PORT=8000

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/generated_tests && \
    chown -R appuser:appuser /app

# Copy project files (honoring .dockerignore)
COPY --chown=appuser:appuser . /app

# Install package with demo extras (ensures pytest is available for validate_test)
RUN pip install --no-cache-dir ".[demo]"

# Seed database at build time
RUN python scripts/seed_demo.py --db /app/agentreplay.db --force && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/healthz || exit 1

# On startup, seed database if file is missing/empty, then launch uvicorn
CMD ["sh", "-c", "if [ ! -s /app/agentreplay.db ]; then python scripts/seed_demo.py --db /app/agentreplay.db --force; fi && exec uvicorn agentreplay.api:app --host 0.0.0.0 --port ${PORT}"]
