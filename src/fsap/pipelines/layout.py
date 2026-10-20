"""Where each layer lives. Locally a directory; on Databricks a Unity Catalog volume or S3 prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: str

    @classmethod
    def from_env(cls) -> Layout:
        return cls(os.environ.get("FSAP_DATA_ROOT", "data"))

    def _p(self, *parts: str) -> str:
        return "/".join([self.root.rstrip("/"), *parts])

    @property
    def landing(self) -> str:
        return self._p("landing")

    def bronze(self, dataset: str) -> str:
        return self._p("bronze", dataset)

    def silver(self, dataset: str) -> str:
        return self._p("silver", dataset)

    def quarantine(self, dataset: str) -> str:
        return self._p("quarantine", dataset)

    def gold(self, name: str) -> str:
        return self._p("gold", name)

    @property
    def reports(self) -> str:
        return self._p("reports")

    @property
    def truth(self) -> str:
        return self._p("truth")

    @property
    def warehouse(self) -> str:
        return self._p("warehouse.duckdb")

    def local(self, path: str) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
