"""Contract enforcement for Spark DataFrames (bronze -> silver in the Databricks jobs)."""

from __future__ import annotations

from collections import Counter

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from fsap.contracts.model import Contract, ContractBreach
from fsap.contracts.pandas_engine import build_report

SPARK_TYPES = {"string": "string", "int": "int", "double": "double", "date": "date", "bool": "boolean"}


def enforce(df: DataFrame, contract: Contract, refs: dict[str, DataFrame] | None = None, raise_on_breach: bool = True):
    """Returns (valid DataFrame typed per contract, quarantine DataFrame with `violations`, report).

    The input keeps its row order in `_row` so that the first occurrence of a duplicated key is the
    one that survives, as in the pandas engine.
    """
    refs = refs or {}
    df = df.withColumn("_row", F.monotonically_increasing_id())
    checks = []
    typed_cols = []
    for f in contract.fields:
        if f.name not in df.columns:
            df = df.withColumn(f.name, F.lit(None).cast("string"))
            checks.append(F.lit(f"missing_column:{f.name}"))
        src = F.trim(F.col(f.name).cast("string"))
        present = F.col(f.name).isNotNull() & (src != F.lit(""))
        if f.type == "int":
            # "12.0" is not an int in the contract; Spark would truncate it, so check it is whole first
            as_double = F.expr(f"try_cast(trim(cast(`{f.name}` as string)) as double)")
            typed = F.when(as_double == F.floor(as_double), as_double.cast("int"))
        else:
            typed = F.when(present, F.expr(f"try_cast(trim(cast(`{f.name}` as string)) as {SPARK_TYPES[f.type]})"))
        tname = f"__t_{f.name}"
        df = df.withColumn(tname, F.when(present, typed))
        typed_cols.append((f.name, tname))
        t = F.col(tname)
        checks.append(F.when(present & t.isNull(), F.lit(f"type:{f.name}")))
        if f.required:
            checks.append(F.when(~present, F.lit(f"required:{f.name}")))
        if f.min is not None:
            checks.append(F.when(t.isNotNull() & (t.cast("double") < f.min), F.lit(f"min:{f.name}")))
        if f.max is not None:
            checks.append(F.when(t.isNotNull() & (t.cast("double") > f.max), F.lit(f"max:{f.name}")))
        if f.enum:
            checks.append(F.when(t.isNotNull() & ~t.isin(list(f.enum)), F.lit(f"enum:{f.name}")))
        if f.pattern:
            checks.append(F.when(t.isNotNull() & ~t.cast("string").rlike(f.pattern), F.lit(f"pattern:{f.name}")))

    for rule in contract.rules:
        left, op, right = rule.parts
        rhs = F.col(f"__t_{right}") if right in contract.columns else F.lit(float(right))
        lhs = F.col(f"__t_{left}")
        cmp = {"<=": lhs <= rhs, ">=": lhs >= rhs, "<": lhs < rhs, ">": lhs > rhs, "==": lhs == rhs, "!=": lhs != rhs}[
            op
        ]
        checks.append(F.when(lhs.isNotNull() & rhs.isNotNull() & ~cmp, F.lit(f"rule:{rule.name}")))

    for ref in contract.references:
        if ref.dataset not in refs:
            continue
        keys = (
            refs[ref.dataset]
            .select(F.col(ref.key).alias(f"__ref_{ref.column}"))
            .distinct()
            .withColumn(f"__ok_{ref.column}", F.lit(True))
        )
        df = df.join(F.broadcast(keys), F.col(f"__t_{ref.column}") == F.col(f"__ref_{ref.column}"), "left")
        checks.append(
            F.when(
                F.col(f"__t_{ref.column}").isNotNull() & F.col(f"__ok_{ref.column}").isNull(),
                F.lit(f"reference:{ref.column}"),
            )
        )

    pk = [F.col(f"__t_{k}") for k in contract.primary_key]
    df = df.withColumn("__dup", F.row_number().over(Window.partitionBy(*pk).orderBy("_row")) > 1)
    checks.append(F.when(F.col("__dup"), F.lit("duplicate_key")))

    df = df.withColumn("violations", F.filter(F.array(*checks), lambda x: x.isNotNull())).cache()
    valid = df.where(F.size("violations") == 0).orderBy("_row").select(*[F.col(t).alias(n) for n, t in typed_cols])
    original = [c for c in df.columns if not c.startswith("__") and c not in ("violations", "_row")]
    quarantine = (
        df.where(F.size("violations") > 0)
        .orderBy("_row")
        .select(
            *[F.col(c).cast("string").alias(c) for c in original], F.concat_ws(";", "violations").alias("violations")
        )
    )
    rows = df.count()
    counts = Counter(
        {
            r["reason"]: r["n"]
            for r in df.select(F.explode("violations").alias("reason"))
            .groupBy("reason")
            .agg(F.count("*").alias("n"))
            .collect()
        }
    )
    n_bad = df.where(F.size("violations") > 0).count()
    report = build_report(contract, rows, counts, n_bad)
    if raise_on_breach and report["breached"]:
        raise ContractBreach(report)
    return valid, quarantine, report
