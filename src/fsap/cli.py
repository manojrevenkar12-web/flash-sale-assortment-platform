"""Command-line entry point; every Airflow task calls one of these commands."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fsap.pipelines.layout import Layout

log = logging.getLogger("fsap")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_simulate(a) -> None:
    from fsap.simulate.generate import generate
    from fsap.simulate.landing import write_landing
    from fsap.simulate.world import WorldConfig

    cfg = WorldConfig(seed=a.seed, days=a.days)
    _print(write_landing(generate(cfg, plan_weeks=a.plan_weeks), Path(a.data)))


def cmd_check_delivery(a) -> None:
    """Fail (so the orchestrator retries later) until this week's delivery is complete."""
    layout = Layout(a.data)
    marker = Path(layout.landing) / "_READY"
    if not marker.exists():
        log.error("landing zone has no _READY marker yet: %s", marker)
        raise SystemExit(4)
    info = json.loads(marker.read_text())
    _print({"delivery": info})


def cmd_ingest(a) -> None:
    from fsap.contracts.model import ContractBreach
    from fsap.pipelines.spark_jobs import ingest, spark_session

    try:
        reports = ingest(spark_session("fsap-ingest"), Layout(a.data))
    except ContractBreach as e:
        log.error("contract breach, run stopped: %s", e)
        _print(e.report)
        raise SystemExit(3) from e
    _print(
        {k: {x: v[x] for x in ("rows", "quarantined", "quarantine_ratio", "top_reasons")} for k, v in reports.items()}
    )


def cmd_features(a) -> None:
    from fsap.pipelines.spark_jobs import features, spark_session

    out = features(spark_session("fsap-features"), Layout(a.data))
    out.pop("profile")
    _print(out)


def cmd_train(a) -> None:
    from fsap.forecasting.train import train
    from fsap.registry.registry import Registry

    out = train(Layout(a.data), rounds=a.rounds, n_folds=a.folds)
    _print({k: out[k] for k in ("version", "promoted")} | {"gate": out["gate"]})
    if not out["promoted"]:
        champ = Registry(a.data).champion()
        if champ is None:
            log.error("the gate rejected the first model and there is no champion to fall back to")
            raise SystemExit(5)
        log.warning(
            "gate rejected version %s; planning continues with champion version %s", out["version"], champ.version
        )


def cmd_plan(a) -> None:
    from fsap.optimisation.job import plan

    _print(plan(Layout(a.data), a.rules, a.campaign or None))


def cmd_evaluate(a) -> None:
    from fsap.optimisation.job import evaluate_policies

    _print(evaluate_policies(Layout(a.data), a.rules, a.limit))


def cmd_curate(a) -> None:
    from fsap.genai.job import curate

    r = curate(Layout(a.data))
    _print(
        {k: r[k] for k in ("extractor", "articles_curated", "copies", "copy_violation_rate", "copy_sources")}
        | {"validator_recall": r["validator"]["recall"]}
    )


def cmd_monitor(a) -> None:
    from fsap.monitoring.monitor import monitor

    r = monitor(Layout(a.data), a.rules)
    _print({"alerts": r["alerts"], "outcomes": r["outcomes"], "checks": len(r["checks"])})
    if r["alerts"] and a.fail_on_alert:
        raise SystemExit(3)


COMMANDS = {
    "monitor": (
        "data, drift and outcome checks with thresholds",
        cmd_monitor,
        [
            (("--rules",), {"default": "conf/planning_rules.yaml"}),
            (("--fail-on-alert",), {"action": "store_true"}),
        ],
    ),
    "curate": ("GenAI curation: attributes and campaign copy, with evaluation", cmd_curate, []),
    "train": (
        "backtest, gate, fit and register the sell-through forecaster",
        cmd_train,
        [
            (("--rounds",), {"type": int, "default": None}),
            (("--folds",), {"type": int, "default": 4}),
        ],
    ),
    "plan": (
        "plan the upcoming campaigns with the champion model",
        cmd_plan,
        [
            (("--rules",), {"default": "conf/planning_rules.yaml"}),
            (("--campaign",), {"action": "append", "help": "plan only these campaign ids"}),
        ],
    ),
    "evaluate": (
        "compare planning policies on the upcoming campaigns against true demand (simulation only)",
        cmd_evaluate,
        [
            (("--rules",), {"default": "conf/planning_rules.yaml"}),
            (("--limit",), {"type": int, "default": None}),
        ],
    ),
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="fsap")
    p.add_argument("--data", default=Layout.from_env().root, help="data root (local dir, dbfs:/ or s3://)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="generate the synthetic outlet and write the landing zone")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--days", type=int, default=730 + 84)
    s.add_argument("--plan-weeks", type=int, default=12)
    s.set_defaults(fn=cmd_simulate)
    sub.add_parser("check-delivery", help="succeed only when the landing zone is complete").set_defaults(
        fn=cmd_check_delivery
    )
    sub.add_parser("ingest", help="landing -> bronze -> silver with contract enforcement").set_defaults(fn=cmd_ingest)
    sub.add_parser("features", help="silver -> gold point-in-time features").set_defaults(fn=cmd_features)
    for name, (help_, fn, extra) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_)
        for args, kwargs in extra:
            sp.add_argument(*args, **kwargs)
        sp.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
