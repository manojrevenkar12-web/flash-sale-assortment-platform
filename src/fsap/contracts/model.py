"""Data contracts: one YAML per dataset, enforced the same way in Spark (pipelines) and pandas (API).

A contract names its owner and version, the fields with types and constraints, row-level rules,
the primary key and foreign keys, and how much bad data a delivery may contain before the whole
run is stopped. Rows that break the contract are not dropped silently: they go to a quarantine
table with the reasons, and the run report counts them per reason.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

TYPES = {"string", "int", "double", "date", "bool"}
_RULE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*(<=|>=|<|>|==|!=)\s*([a-z_][a-z0-9_]*|-?[0-9.]+)\s*$")
OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq, "!=": operator.ne}


class ContractError(ValueError):
    """The contract file itself is invalid."""


class ContractBreach(RuntimeError):
    """A delivery broke its contract badly enough that the run must stop."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"{report['dataset']}: {report['quarantined']} of {report['rows']} rows quarantined "
            f"({report['quarantine_ratio']:.2%} > {report['max_quarantine_ratio']:.2%}); top reasons {report['top_reasons']}"
        )


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    required: bool = True
    min: float | None = None
    max: float | None = None
    enum: tuple[str, ...] | None = None
    pattern: str | None = None


@dataclass(frozen=True)
class Rule:
    name: str
    expr: str

    @property
    def parts(self) -> tuple[str, str, str]:
        m = _RULE.match(self.expr)
        assert m, self.expr
        return m.group(1), m.group(2), m.group(3)


@dataclass(frozen=True)
class Reference:
    column: str
    dataset: str
    key: str


@dataclass(frozen=True)
class Contract:
    name: str
    version: str
    owner: str
    description: str
    primary_key: tuple[str, ...]
    max_quarantine_ratio: float
    fields: tuple[Field, ...]
    rules: tuple[Rule, ...] = ()
    references: tuple[Reference, ...] = ()
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self.fields]

    def field(self, name: str) -> Field:
        return next(f for f in self.fields if f.name == name)


def parse(doc: dict) -> Contract:
    try:
        fields = tuple(
            Field(
                name=f["name"],
                type=f["type"],
                required=bool(f.get("required", True)),
                min=f.get("min"),
                max=f.get("max"),
                enum=tuple(f["enum"]) if f.get("enum") else None,
                pattern=f.get("pattern"),
            )
            for f in doc["fields"]
        )
        c = Contract(
            name=doc["name"],
            version=str(doc["version"]),
            owner=doc["owner"],
            description=str(doc.get("description", "")).strip(),
            primary_key=tuple(doc["primary_key"]),
            max_quarantine_ratio=float(doc["max_quarantine_ratio"]),
            fields=fields,
            rules=tuple(Rule(**r) for r in doc.get("rules") or []),
            references=tuple(Reference(**r) for r in doc.get("references") or []),
            raw=doc,
        )
    except (KeyError, TypeError) as e:
        raise ContractError(f"invalid contract: {e}") from e
    names = c.columns
    if len(set(names)) != len(names):
        raise ContractError(f"{c.name}: duplicate field names")
    if not re.fullmatch(r"\d+\.\d+\.\d+", c.version):
        raise ContractError(f"{c.name}: version must be semantic (x.y.z)")
    for f in fields:
        if f.type not in TYPES:
            raise ContractError(f"{c.name}.{f.name}: unknown type {f.type}")
    for k in c.primary_key:
        if k not in names:
            raise ContractError(f"{c.name}: primary key {k} is not a field")
    for r in c.rules:
        m = _RULE.match(r.expr)
        if not m:
            raise ContractError(f"{c.name}: rule {r.name} must be '<column> <op> <column|number>'")
        for side in (m.group(1), m.group(3)):
            if re.fullmatch(r"[a-z_][a-z0-9_]*", side) and side not in names:
                raise ContractError(f"{c.name}: rule {r.name} refers to unknown column {side}")
    return c


def load(name: str) -> Contract:
    text = resources.files("fsap.contracts").joinpath(f"{name}.yaml").read_text()
    return parse(yaml.safe_load(text))


def load_file(path: Path) -> Contract:
    return parse(yaml.safe_load(Path(path).read_text()))


def all_contracts() -> dict[str, Contract]:
    out = {}
    for p in resources.files("fsap.contracts").iterdir():
        if p.name.endswith(".yaml"):
            c = parse(yaml.safe_load(p.read_text()))
            out[c.name] = c
    return out


def breaking_changes(old: Contract, new: Contract) -> list[str]:
    """Changes that would break consumers of `old`. A breaking change needs a new major version."""
    problems = []
    for f in old.fields:
        if f.name not in new.columns:
            problems.append(f"field {f.name} removed")
            continue
        g = new.field(f.name)
        if g.type != f.type and not (f.type == "int" and g.type == "double"):
            problems.append(f"field {f.name} changed type {f.type} -> {g.type}")
        if f.required and not g.required:
            problems.append(f"field {f.name} became optional (consumers rely on it being present)")
        if f.enum and g.enum and set(g.enum) - set(f.enum):
            problems.append(f"field {f.name} accepts new values {sorted(set(g.enum) - set(f.enum))}")
    if tuple(old.primary_key) != tuple(new.primary_key):
        problems.append("primary key changed")
    return problems


def check_version_bump(old: Contract, new: Contract) -> list[str]:
    """Breaking changes require a major bump; any change requires some bump."""
    o = tuple(int(x) for x in old.version.split("."))
    n = tuple(int(x) for x in new.version.split("."))
    breaking = breaking_changes(old, new)
    if breaking and n[0] <= o[0]:
        return [f"breaking change without a major version bump ({old.version} -> {new.version}): {b}" for b in breaking]
    if old.raw != new.raw and n <= o:
        return [f"contract changed but version {new.version} is not higher than {old.version}"]
    return []
