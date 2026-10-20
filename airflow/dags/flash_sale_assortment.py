"""Daily assortment pipeline for flash-sale campaigns.

check_delivery -> ingest (Spark) -> dbt_build -> features (Spark) -> train -> plan -> curate -> monitor

Where each step runs is decided by FSAP_RUNTIME (an Airflow Variable or environment variable):

- "local"       every step is `python -m fsap.cli ...` in the project's own virtualenv
                (FSAP_PYTHON); used by `airflow dags test` in CI and on laptops
- "production"  Spark steps run as Databricks jobs defined in deploy/databricks/bundle.yml;
                Python steps (training, planning, curation, monitoring, dbt) run as pods from the
                project image on Kubernetes, with resources and secrets set per step

Airflow only orchestrates; it never imports the project, so the scheduler stays light and its
dependencies never clash with the ML stack.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta

from airflow.sdk import DAG, Variable

log = logging.getLogger(__name__)

RUNTIME = os.environ.get("FSAP_RUNTIME") or Variable.get("fsap_runtime", default="local")
DATA_ROOT = os.environ.get("FSAP_DATA_ROOT", "data")
FSAP_PYTHON = os.environ.get("FSAP_PYTHON", "python")
PROJECT_DIR = os.environ.get("FSAP_PROJECT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
IMAGE = os.environ.get("FSAP_IMAGE", "ghcr.io/manojrevenkar12-web/flash-sale-assortment-platform:latest")
NAMESPACE = os.environ.get("FSAP_NAMESPACE", "assortment")


def notify_failure(context) -> None:
    """Post failed tasks to the team channel (webhook from the environment); always logged."""
    ti = context["task_instance"]
    msg = {
        "text": f":red_circle: {ti.dag_id}.{ti.task_id} failed (try {ti.try_number}) for {context.get('logical_date')}. Runbook: docs/runbooks.md#{ti.task_id}",
    }
    log.error(msg["text"])
    url = os.environ.get("FSAP_ALERT_WEBHOOK")
    if url:
        try:
            req = urllib.request.Request(
                url, data=json.dumps(msg).encode(), headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:  # never let alerting break the callback
            log.warning("alert webhook failed: %s", e)


# step -> (engine, timeout, retries, resources)
STEPS = {
    "check_delivery": ("python", timedelta(minutes=5), 6, {"cpu": "250m", "memory": "512Mi"}),
    "ingest": ("spark", timedelta(minutes=45), 0, None),  # a contract breach is not fixed by retrying
    "dbt_build": ("dbt", timedelta(minutes=30), 1, {"cpu": "1", "memory": "2Gi"}),
    "features": ("spark", timedelta(minutes=45), 1, None),
    "train": ("python", timedelta(hours=2), 1, {"cpu": "4", "memory": "12Gi"}),
    "plan": ("python", timedelta(hours=1), 1, {"cpu": "2", "memory": "6Gi"}),
    "curate": ("python", timedelta(minutes=45), 2, {"cpu": "1", "memory": "4Gi"}),
    "monitor": ("python", timedelta(minutes=20), 1, {"cpu": "1", "memory": "4Gi"}),
}
CLI = {
    "check_delivery": ["check-delivery"],
    "ingest": ["ingest"],
    "features": ["features"],
    "train": ["train"],
    "plan": ["plan"],
    "curate": ["curate"],
    "monitor": ["monitor"],
}


def make_task(step: str):
    engine, timeout, retries, resources = STEPS[step]
    common = {
        "task_id": step,
        "execution_timeout": timeout,
        "retries": retries,
        "on_failure_callback": notify_failure,
    }
    if RUNTIME == "local":
        from airflow.providers.standard.operators.bash import BashOperator

        if engine == "dbt":
            cmd = f"cd {PROJECT_DIR}/dbt && FSAP_DATA_ROOT={os.path.abspath(DATA_ROOT)} {os.path.dirname(FSAP_PYTHON)}/dbt build --profiles-dir . --target local"
        else:
            cmd = f"cd {PROJECT_DIR} && {FSAP_PYTHON} -m fsap.cli --data {DATA_ROOT} " + " ".join(CLI[step])
        return BashOperator(bash_command=cmd, **common)
    if engine == "spark":
        from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator

        return DatabricksRunNowOperator(databricks_conn_id="databricks_default", job_name=f"fsap-{step}", **common)
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    from kubernetes.client import models as k8s

    args = (
        ["dbt", "build", "--project-dir", "/app/dbt", "--profiles-dir", "/app/dbt", "--target", "prod"]
        if engine == "dbt"
        else ["fsap", *CLI[step]]
    )
    return KubernetesPodOperator(
        name=f"fsap-{step.replace('_', '-')}",
        namespace=NAMESPACE,
        image=IMAGE,
        cmds=[],
        arguments=args,
        service_account_name="fsap-batch",
        env_from=[
            k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="fsap-batch")),
            k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name="fsap-config")),
        ],
        container_resources=k8s.V1ResourceRequirements(requests=resources, limits={"memory": resources["memory"]}),
        security_context=k8s.V1PodSecurityContext(
            run_as_non_root=True, run_as_user=10001, seccomp_profile=k8s.V1SeccompProfile(type="RuntimeDefault")
        ),
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
        **common,
    )


with DAG(
    dag_id="flash_sale_assortment",
    description="Contracts, dbt, features, training with a release gate, campaign plans, GenAI curation, monitoring",
    schedule="30 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=5),
    default_args={
        "owner": "lounge-assortment-ml",
        "retry_delay": timedelta(minutes=10),
        "retry_exponential_backoff": True,
    },
    tags=["assortment", "forecasting", "optimisation", "genai"],
    doc_md=__doc__,
) as dag:
    tasks = {s: make_task(s) for s in STEPS}
    (
        tasks["check_delivery"]
        >> tasks["ingest"]
        >> tasks["dbt_build"]
        >> tasks["features"]
        >> tasks["train"]
        >> tasks["plan"]
        >> tasks["curate"]
        >> tasks["monitor"]
    )
