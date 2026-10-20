"""DAG integrity, run in the Airflow environment (not the project's): the DAG imports cleanly in
both runtimes, steps run in the right order, and every step has a timeout, an owner and a
failure alert."""

import importlib
import os
import sys
from pathlib import Path

import pytest

DAGS = Path(__file__).resolve().parents[1] / "dags"
ORDER = ["check_delivery", "ingest", "dbt_build", "features", "train", "plan", "curate", "monitor"]


def load(runtime: str):
    os.environ["FSAP_RUNTIME"] = runtime
    sys.path.insert(0, str(DAGS))
    sys.modules.pop("flash_sale_assortment", None)
    mod = importlib.import_module("flash_sale_assortment")
    return mod.dag


@pytest.mark.parametrize("runtime", ["local", "production"])
def test_dag_imports_and_is_ordered(runtime):
    dag = load(runtime)
    assert [t.task_id for t in dag.topological_sort()] == ORDER
    for up, down in zip(ORDER, ORDER[1:], strict=False):
        assert down in dag.get_task(up).downstream_task_ids


@pytest.mark.parametrize("runtime", ["local", "production"])
def test_every_task_is_operable(runtime):
    dag = load(runtime)
    assert dag.max_active_runs == 1 and not dag.catchup and dag.dagrun_timeout
    for t in dag.tasks:
        assert t.execution_timeout is not None, t.task_id
        assert t.owner == "lounge-assortment-ml"
        assert t.on_failure_callback, t.task_id
    assert dag.get_task("ingest").retries == 0  # bad data is not fixed by retrying


def test_production_uses_databricks_for_spark_and_pods_for_the_rest():
    dag = load("production")
    kinds = {t.task_id: type(t).__name__ for t in dag.tasks}
    assert kinds["ingest"] == kinds["features"] == "DatabricksRunNowOperator"
    assert all(kinds[s] == "KubernetesPodOperator" for s in ("train", "plan", "curate", "monitor", "dbt_build"))
    train = dag.get_task("train")
    assert train.arguments == ["fsap", "train"] and train.security_context.run_as_non_root
