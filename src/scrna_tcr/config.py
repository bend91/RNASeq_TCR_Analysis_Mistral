"""Config loading and small shared helpers."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger("scrna_tcr")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dirs(cfg: dict) -> dict:
    base = ensure_dir(cfg["output"]["results_dir"])
    d = {
        "base": base,
        "tables": ensure_dir(base / "tables"),
        "figures": ensure_dir(base / "figures"),
        "processed": ensure_dir(base / "processed"),
    }
    return d
