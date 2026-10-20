# syntax=docker/dockerfile:1.7
# runtime: the API and the batch steps that run as Kubernetes pods (training, planning, curation,
#          monitoring, dbt). Spark steps run on Databricks, so no JVM here.
# local:   adds a JRE so docker compose can run the whole pipeline, Spark included, on one machine.
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml ./
COPY src/ src/
COPY conf/ conf/
COPY dbt/ dbt/
RUN pip install --no-deps . && mkdir -p /data && chown -R 10001:10001 /app /data
USER 10001
ENV FSAP_DATA_ROOT=/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "fsap.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

FROM runtime AS local
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
COPY scripts/ scripts/
USER 10001
