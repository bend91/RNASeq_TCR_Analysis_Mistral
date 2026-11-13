"""Loading of scRNA-seq (10x) and TCR-seq (AIRR / CellRanger VDJ) data.

Every sample's cell barcodes are prefixed with ``<sample_id>_`` so that
GEX cells and TCR cells can be matched and clonotypes defined globally
across all samples (required for lymph-node / tumour sharing analysis).
"""
from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import scirpy as ir

from .access import make_mdata

log = logging.getLogger("scrna_tcr.io")


def load_sample_sheet(path: str | Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = {"sample_id", "patient", "treatment", "tissue"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"sample metadata is missing columns: {missing}")
    return meta


def load_scrna_sample(sample_id: str, scrna_dir: str | Path, fmt: str) -> ad.AnnData:
    """Load one scRNA-seq sample; obs names become ``<sample_id>_<barcode>``."""
    if fmt == "h5":
        candidates = list(Path(scrna_dir).glob(f"{sample_id}*.h5"))
        if not candidates:
            raise FileNotFoundError(f"no h5 found for sample {sample_id} in {scrna_dir}")
        adata = sc.read_10x_h5(candidates[0])
    elif fmt == "mtx":
        candidates = [p for p in Path(scrna_dir).iterdir() if p.is_dir() and p.name.startswith(sample_id)]
        if not candidates:
            raise FileNotFoundError(f"no 10x mtx dir found for sample {sample_id} in {scrna_dir}")
        adata = sc.read_10x_mtx(candidates[0], var_names="gene_symbols", cache=False)
    else:
        raise ValueError(f"unknown scrna_format: {fmt}")

    adata.var_names_make_unique()
    adata.obs_names = [f"{sample_id}_{b.split('-')[0]}" for b in adata.obs_names]
    adata.obs["sample_id"] = sample_id
    adata.X = adata.X.astype("float32")
    return adata


def _tcr_path(tcr_dir: str | Path, sample_id: str, fmt: str) -> Path:
    suffixes = [".tsv", ".tsv.gz", ".csv"] if fmt == "airr" else [".csv"]
    for suf in suffixes:
        cand = Path(tcr_dir) / f"{sample_id}{suf}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no TCR file for sample {sample_id} ({fmt}) in {tcr_dir}")


def _prefix_cell_ids(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    df = df.copy()
    if "cell_id" not in df.columns:
        raise ValueError("TCR table must contain a 'cell_id' column")
    df["cell_id"] = [f"{sample_id}_{str(c).split('-')[0]}" for c in df["cell_id"]]
    if "is_cell" not in df.columns:
        df["is_cell"] = True
    return df


def load_tcr_sample(sample_id: str, tcr_dir: str | Path, fmt: str) -> ad.AnnData:
    """Load one sample's TCR data as a chain AnnData (scirpy format)."""
    path = _tcr_path(tcr_dir, sample_id, fmt)
    if fmt == "airr":
        sep = "\t" if str(path).endswith((".tsv", ".tsv.gz")) else ","
        df = pd.read_csv(path, sep=sep)
        df = _prefix_cell_ids(df, sample_id)
        return ir.io.read_airr(df)
    elif fmt == "tenx":
        df = pd.read_csv(path)
        df = df.rename(columns={"barcodes": "cell_id"})
        df = _prefix_cell_ids(df, sample_id)
        return ir.io.read_10x_vdj(df)
    raise ValueError(f"unknown tcr_format: {fmt}")


def load_dataset(cfg: dict):
    """Load all samples; returns (gex_adata, ir_adata_or_None)."""
    meta = load_sample_sheet(cfg["data"]["sample_metadata"])
    fmt_gex = cfg["data"]["scrna_format"]
    fmt_tcr = cfg["data"]["tcr_format"]

    gex_parts, ir_parts = [], []
    for _, row in meta.iterrows():
        sid = row["sample_id"]
        log.info("loading sample %s (%s / %s)", sid, row["treatment"], row["tissue"])
        a = load_scrna_sample(sid, cfg["data"]["scrna_dir"], fmt_gex)
        a.obs["patient"] = str(row["patient"])
        a.obs["treatment"] = str(row["treatment"])
        a.obs["tissue"] = str(row["tissue"])
        gex_parts.append(a)
        try:
            ir_parts.append(load_tcr_sample(sid, cfg["data"]["tcr_dir"], fmt_tcr))
        except FileNotFoundError:
            log.warning("no TCR data for sample %s - continuing without it", sid)

    gex = ad.concat(gex_parts, join="outer", label=None)
    gex.obs_names_make_unique()

    airr = None
    if ir_parts:
        airr = ir_parts[0]
        for extra in ir_parts[1:]:
            airr = ir.pp.merge_airr(airr, extra)
        log.info("TCR chains loaded for %d cells", airr.n_obs)
    return gex, airr


def merge_gex_with_ir(gex: ad.AnnData, airr: ad.AnnData | None):
    """Build the MuData object used by the rest of the pipeline."""
    if airr is None:
        log.warning("no TCR data available - running GEX-only")
    mdata = make_mdata(gex, airr)
    return mdata
