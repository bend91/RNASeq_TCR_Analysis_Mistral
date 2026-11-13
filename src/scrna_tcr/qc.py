"""Quality control of the scRNA-seq component."""
from __future__ import annotations

import logging

import anndata as ad
import scanpy as sc

log = logging.getLogger("scrna_tcr.qc")


def compute_qc_metrics(adata: ad.AnnData) -> None:
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )


def filter_cells(adata: ad.AnnData, qc_cfg: dict) -> ad.AnnData:
    n0 = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=qc_cfg["min_genes"])
    sc.pp.filter_cells(adata, min_counts=qc_cfg["min_counts"])
    sc.pp.filter_genes(adata, min_cells=3)
    if "pct_counts_mt" in adata.obs:
        adata = adata[adata.obs["pct_counts_mt"] < qc_cfg["max_pct_mito"], :].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    log.info("QC: %d -> %d cells, %d genes", n0, adata.n_obs, adata.n_vars)
    return adata


def detect_doublets(adata: ad.AnnData) -> None:
    """Scrublet doublet scores; silently skipped if scrublet is unavailable."""
    try:
        import scrublet as scr
    except ImportError:
        log.warning("scrublet not installed - skipping doublet detection")
        return
    for sample, idx in adata.obs.groupby("sample_id", observed=True).groups.items():
        sub = adata[idx]
        try:
            scores = scr.scrublet(sub.X)[0]
            adata.obs.loc[idx, "doublet_score"] = scores
        except Exception as exc:  # noqa: BLE001
            log.warning("scrublet failed for %s: %s", sample, exc)
    if "doublet_score" in adata.obs:
        adata.obs["predicted_doublet"] = adata.obs["doublet_score"] > 0.5
        n = int(adata.obs["predicted_doublet"].sum())
        log.info("predicted %d doublets (%.1f%%)", n, 100 * n / adata.n_obs)


def run_qc(adata: ad.AnnData, qc_cfg: dict) -> ad.AnnData:
    compute_qc_metrics(adata)
    adata = filter_cells(adata, qc_cfg)
    if qc_cfg.get("doublet_detection", True):
        detect_doublets(adata)
        if "predicted_doublet" in adata.obs:
            adata = adata[~adata.obs["predicted_doublet"], :].copy()
    return adata
