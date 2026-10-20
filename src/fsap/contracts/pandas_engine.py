"""Contract enforcement for pandas frames (API payloads, small batches, tests).

Semantics match the Spark engine exactly; a test runs both on the same dirty data.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from fsap.contracts.model import OPS, Contract, ContractBreach


def _cast(s: pd.Series, typ: str) -> pd.Series:
    if typ == "string":
        return s.where(s.isna(), s.astype(str))
    if typ in ("int", "double"):
        num = pd.to_numeric(s, errors="coerce")
        if typ == "int":
            whole = num.notna() & (num % 1 == 0)
            return num.where(whole).astype("Int64")
        return num.astype("Float64")
    if typ == "date":
        return pd.to_datetime(s, errors="coerce", format="mixed").dt.date
    if typ == "bool":
        truthy = {"true": True, "1": True, "false": False, "0": False}
        return s.map(lambda v: truthy.get(str(v).strip().lower()) if v is not None else None)
    raise ValueError(typ)


def enforce(df: pd.DataFrame, contract: Contract, refs: dict[str, set] | None = None, raise_on_breach: bool = True):
    """Returns (valid rows typed per contract, quarantined rows with a `violations` column, report)."""
    refs = refs or {}
    raw = df.reset_index(drop=True)
    out = pd.DataFrame(index=raw.index)
    reasons: list[list[str]] = [[] for _ in range(len(raw))]

    def flag(mask: np.ndarray, reason: str):
        for i in np.flatnonzero(mask):
            reasons[i].append(reason)

    for f in contract.fields:
        if f.name not in raw.columns:
            flag(np.ones(len(raw), bool), f"missing_column:{f.name}")
            out[f.name] = pd.Series([None] * len(raw), dtype=object)
            continue
        src = raw[f.name]
        present = src.notna() & (src.astype(str).str.strip() != "")
        typed = _cast(src.where(present), f.type)
        out[f.name] = typed
        flag((present & typed.isna()).to_numpy(), f"type:{f.name}")
        if f.required:
            flag((~present).to_numpy(), f"required:{f.name}")
        ok = typed.notna()
        if f.min is not None:
            flag((ok & (typed.astype("Float64") < f.min)).fillna(False).to_numpy(bool), f"min:{f.name}")
        if f.max is not None:
            flag((ok & (typed.astype("Float64") > f.max)).fillna(False).to_numpy(bool), f"max:{f.name}")
        if f.enum:
            flag((ok & ~typed.isin(f.enum)).to_numpy(bool), f"enum:{f.name}")
        if f.pattern:
            flag((ok & ~typed.astype(str).str.fullmatch(f.pattern)).to_numpy(bool), f"pattern:{f.name}")

    for rule in contract.rules:
        left, op, right = rule.parts
        lv = out[left]
        rv = out[right] if right in out.columns else float(right)
        both = lv.notna() & (rv.notna() if isinstance(rv, pd.Series) else True)
        res = OPS[op](lv, rv)
        flag((both & ~res.fillna(True).astype(bool)).to_numpy(bool), f"rule:{rule.name}")

    for ref in contract.references:
        if ref.dataset in refs:
            known = refs[ref.dataset]
            flag((out[ref.column].notna() & ~out[ref.column].isin(known)).to_numpy(bool), f"reference:{ref.column}")

    pk = list(contract.primary_key)
    dup = out.duplicated(subset=pk, keep="first").to_numpy()
    flag(dup, "duplicate_key")

    bad = np.array([bool(r) for r in reasons])
    valid = out[~bad].reset_index(drop=True)
    quarantine = raw[bad].copy()
    quarantine["violations"] = [";".join(r) for r, b in zip(reasons, bad, strict=True) if b]
    report = build_report(contract, len(raw), Counter(x for r in reasons for x in r), int(bad.sum()))
    if raise_on_breach and report["breached"]:
        raise ContractBreach(report)
    return valid, quarantine.reset_index(drop=True), report


def build_report(contract: Contract, rows: int, counts: Counter, quarantined: int) -> dict:
    ratio = quarantined / rows if rows else 0.0
    return {
        "dataset": contract.name,
        "contract_version": contract.version,
        "rows": rows,
        "valid": rows - quarantined,
        "quarantined": quarantined,
        "quarantine_ratio": ratio,
        "max_quarantine_ratio": contract.max_quarantine_ratio,
        "breached": ratio > contract.max_quarantine_ratio,
        "reasons": dict(sorted(counts.items())),
        "top_reasons": [k for k, _ in counts.most_common(3)],
    }
