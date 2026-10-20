"""Training job: backtest, gate, fit on all history, register.

Two paths through the gate:
- a new recipe (different model or parameters) must beat the current champion's recipe
  significantly, fold by fold, on the same backtest;
- a refresh (same recipe, newer data) must stay unbiased and calibrated and must not be more
  than `refresh_tolerance` worse than the WAPE recorded for the champion.
With no champion yet, the challenger has to beat the history baseline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from fsap.contracts.model import all_contracts
from fsap.forecasting.backtest import GatePolicy, release_gate, run_backtest, summarise
from fsap.forecasting.model import FEATURES, GbmForecaster, HistoryBaseline, frame_fingerprint
from fsap.pipelines.layout import Layout
from fsap.registry.registry import Registry

log = logging.getLogger(__name__)


def recipe_hash(model: GbmForecaster) -> str:
    return hashlib.sha256(
        json.dumps(
            {"name": model.name, "params": model.params, "rounds": model.rounds, "features": FEATURES}, sort_keys=True
        ).encode()
    ).hexdigest()[:12]


def train(
    layout: Layout, rounds: int | None = None, n_folds: int = 4, refresh_tolerance: float = 0.02, use_truth: bool = True
) -> dict:
    frame = pd.read_parquet(layout.gold("training_features"))
    truth_path = Path(layout.truth) / "offer_truth.parquet"
    truth = pd.read_parquet(truth_path) if use_truth and truth_path.exists() else None
    reg = Registry(layout.root)
    champ = reg.champion()

    challenger_factory = lambda: GbmForecaster(rounds=rounds)  # noqa: E731
    candidate = challenger_factory()
    recipe = recipe_hash(candidate)
    refresh = champ is not None and champ.tags.get("recipe") == recipe
    if champ is None:
        champion_name, champion_factory = "history_baseline", HistoryBaseline
    elif refresh:
        champion_name, champion_factory = "history_baseline", HistoryBaseline  # reference only
    else:
        champion_name = f"champion_v{champ.version}"
        champion_factory = lambda: GbmForecaster(rounds=champ.model.rounds)  # noqa: E731

    results = run_backtest(
        frame, {"challenger": challenger_factory, champion_name: champion_factory}, truth=truth, n_folds=n_folds
    )
    summary = summarise(results)
    gate = release_gate(results, "challenger", champion_name, GatePolicy())
    ch_wape = float(summary.loc[summary["model"] == "challenger", "wape"].mean())
    if refresh:
        recorded = float(champ.tags.get("backtest_wape", "nan"))
        gate["path"] = "refresh"
        gate["checks"]["significantly_better"] = True  # not required for a refresh
        gate["checks"]["no_worse_than_champion"] = not (ch_wape > recorded * (1 + refresh_tolerance))
        gate["promote"] = all(gate["checks"].values())
    else:
        gate["path"] = "new_recipe" if champ else "first_model"

    # final model: everything up to two weeks ago for fitting, the last two weeks for calibration
    end = pd.to_datetime(frame["end_date"])
    cut = end.max() - pd.Timedelta(days=14)
    model = challenger_factory().fit(frame[end < cut])
    model.calibrate(frame[end >= cut])
    contracts = {n: c.version for n, c in all_contracts().items()}
    fingerprint = frame_fingerprint(frame, ["campaign_id", "article_id", "demand_est"])
    model.meta = {
        "trained_on_rows": len(frame),
        "data_fingerprint": fingerprint,
        "contracts": contracts,
        "trained_until": str(end.max().date()),
    }

    metrics = {"backtest_wape": ch_wape}
    for col in ("bias", "coverage80_uncensored", "wape_true", "coverage80_true"):
        if col in summary:
            metrics[f"backtest_{col}"] = float(summary.loc[summary["model"] == "challenger", col].mean())
    metrics["gate_wape_difference"] = gate["wape_difference"]["difference"]
    tags = {
        "recipe": recipe,
        "data_fingerprint": fingerprint,
        "contracts": json.dumps(contracts),
        "gate_path": gate["path"],
        "promoted": gate["promote"],
        "backtest_wape": round(ch_wape, 5),
        "dispersion": round(model.dispersion, 4),
        "level": round(model.level, 4),
    }
    run_id = reg.log_run(
        model,
        params={
            **{k: v for k, v in model.params.items() if k != "verbose"},
            "rounds": model.rounds,
            "n_features": len(FEATURES),
        },
        metrics=metrics,
        tags=tags,
        reports={"backtest.csv": summary.to_csv(index=False), "gate.json": gate, "importance.json": model.importance()},
    )
    version = reg.register(run_id, tags, promote=gate["promote"])
    out = {
        "version": version,
        "run_id": run_id,
        "promoted": gate["promote"],
        "gate": gate,
        "backtest": summary.round(4).to_dict(orient="records"),
    }
    Path(layout.local(f"{layout.reports}/training")).joinpath("latest.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    log.info("model version %s %s", version, "promoted to champion" if gate["promote"] else "rejected by the gate")
    return out
