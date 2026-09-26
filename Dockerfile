# Slim, stateless API runtime serving the model artifacts produced by `python ml/train.py`
# (artifacts/ are committed; to re-train inside Docker use Dockerfile.train).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ARTIFACTS_DIR=/app/artifacts STREAM_DIR=/data/stream WEB_CONCURRENCY=2
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt && useradd -m app && mkdir -p /data/stream && chown app /data/stream
COPY backend ./backend
COPY artifacts/cube.parquet artifacts/meta.json artifacts/stops.json artifacts/intervals.json artifacts/pf_particles.npz ./artifacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
# one uvicorn worker per vCPU; no shared in-process state -> safe to scale out
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --no-access-log --loop uvloop --http httptools"]
