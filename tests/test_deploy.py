"""Deployment files stay consistent with the code."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_cluster_alerts_match_the_compose_alerts():
    compose = yaml.safe_load((ROOT / "ops/prometheus/alerts.yml").read_text())
    cluster = yaml.safe_load((ROOT / "deploy/k8s/base/alerts.yaml").read_text())
    assert cluster["spec"] == compose


def test_every_alert_links_an_existing_runbook_section():
    runbooks = (ROOT / "docs/runbooks.md").read_text().lower()
    anchors = {"#" + line[3:].strip().replace(" ", "-") for line in runbooks.splitlines() if line.startswith("## ")}
    for group in yaml.safe_load((ROOT / "ops/prometheus/alerts.yml").read_text())["groups"]:
        for rule in group["rules"]:
            assert rule["annotations"]["runbook"].split("runbooks.md")[1] in anchors, rule["alert"]


def test_dashboard_only_uses_metrics_the_api_exports():
    from fsap.serving.app import build_metrics

    exported = set()
    for m in build_metrics().values():
        if hasattr(m, "_name"):
            exported.add(m._name)
    dash = json.loads((ROOT / "ops/grafana/dashboards/fsap.json").read_text())
    for p in dash["panels"]:
        for t in p.get("targets", []):
            assert any(name in t["expr"] for name in exported), t["expr"]


def test_databricks_jobs_match_the_dag_spark_steps():
    bundle = yaml.safe_load((ROOT / "deploy/databricks/bundle.yml").read_text())
    assert set(bundle["resources"]["jobs"]) == {"fsap-ingest", "fsap-features"}
    for job in bundle["resources"]["jobs"].values():
        task = job["tasks"][0]["python_wheel_task"]
        assert task["entry_point"] == "fsap" and task["parameters"][-1] in ("ingest", "features")
