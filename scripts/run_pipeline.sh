#!/usr/bin/env bash
# The DAG's steps in order, for docker compose and quick local runs (Airflow does this in production).
set -euo pipefail
DATA="${FSAP_DATA_ROOT:-data}"
DAYS="${FSAP_SIM_DAYS:-814}"
step() { echo "== $*"; }
if [ ! -f "$DATA/landing/_READY" ]; then step simulate; python -m fsap.cli --data "$DATA" simulate --days "$DAYS"; fi
step check-delivery; python -m fsap.cli --data "$DATA" check-delivery
step ingest;         python -m fsap.cli --data "$DATA" ingest
step dbt;            (cd dbt && FSAP_DATA_ROOT="$(cd "../$DATA" 2>/dev/null && pwd || echo "$DATA")" dbt build --profiles-dir . --target local)
step features;       python -m fsap.cli --data "$DATA" features
step train;          python -m fsap.cli --data "$DATA" train
step plan;           python -m fsap.cli --data "$DATA" plan
step curate;         python -m fsap.cli --data "$DATA" curate
step monitor;        python -m fsap.cli --data "$DATA" monitor
