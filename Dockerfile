FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DATA_DIR=/app/data \
    LIBRARY_DB_PATH=/app/data/library.db \
    CHECKPOINT_DB_PATH=/app/data/checkpoints.db

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
COPY src ./src

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /app/data

CMD ["movie-recommender"]
