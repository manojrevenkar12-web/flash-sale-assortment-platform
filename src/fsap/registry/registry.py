"""Model registry on MLflow: every training run is tracked, every model version carries the data
fingerprint, contract versions, feature list and gate decision it was trained with, and the
serving and planning jobs load whatever the `champion` alias points at.

Locally MLflow runs on SQLite under the data root; on Databricks set FSAP_MLFLOW_URI=databricks
and FSAP_MLFLOW_REGISTRY_URI=databricks-uc.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from fsap.forecasting.model import GbmForecaster

MODEL_NAME = "sell_through_forecaster"
EXPERIMENT = "flash-sale-sell-through"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@dataclass
class Champion:
    version: str
    run_id: str
    model: GbmForecaster
    tags: dict


class Registry:
    def __init__(self, data_root: str):
        root = Path(data_root).resolve()
        self.tracking_uri = os.environ.get("FSAP_MLFLOW_URI", f"sqlite:///{root}/mlflow.db")
        self.registry_uri = os.environ.get("FSAP_MLFLOW_REGISTRY_URI", self.tracking_uri)
        self.artifact_root = os.environ.get("FSAP_MLFLOW_ARTIFACTS", (root / "mlruns").as_uri())
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        self.client = MlflowClient(self.tracking_uri, self.registry_uri)
        exp = self.client.get_experiment_by_name(EXPERIMENT)
        self.experiment_id = (
            exp.experiment_id
            if exp
            else self.client.create_experiment(EXPERIMENT, artifact_location=self.artifact_root)
        )

    def log_run(self, model: GbmForecaster, params: dict, metrics: dict, tags: dict, reports: dict[str, object]) -> str:
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "model.json").write_text(json.dumps(model.to_artifact()))
                mlflow.log_artifact(str(Path(d) / "model.json"), "model")
                for name, obj in reports.items():
                    (Path(d) / name).write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str))
                    mlflow.log_artifact(str(Path(d) / name), "reports")
            return run.info.run_id

    def register(self, run_id: str, tags: dict, promote: bool) -> str:
        with contextlib.suppress(MlflowException):  # already registered
            self.client.create_registered_model(
                MODEL_NAME,
                description="Negative-binomial demand forecast per article and campaign (LightGBM, Tweedie).",
            )
        src = f"{self.client.get_run(run_id).info.artifact_uri}/model"
        mv = self.client.create_model_version(MODEL_NAME, src, run_id=run_id, tags={k: str(v) for k, v in tags.items()})
        if promote:
            self.client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        else:
            self.client.set_model_version_tag(MODEL_NAME, mv.version, "rejected_by_gate", "true")
        return mv.version

    def champion(self) -> Champion | None:
        try:
            mv = self.client.get_model_version_by_alias(MODEL_NAME, "champion")
        except MlflowException:
            return None
        path = mlflow.artifacts.download_artifacts(f"{mv.source}/model.json")
        model = GbmForecaster.from_artifact(json.loads(Path(path).read_text()))
        return Champion(mv.version, mv.run_id, model, dict(mv.tags))

    def versions(self) -> list[dict]:
        try:
            vs = self.client.search_model_versions(f"name='{MODEL_NAME}'")
        except MlflowException:
            return []
        champ = self.champion()
        return sorted(
            (
                {
                    "version": v.version,
                    "run_id": v.run_id,
                    "champion": champ is not None and champ.version == v.version,
                    "tags": dict(v.tags),
                }
                for v in vs
            ),
            key=lambda d: int(d["version"]),
        )
