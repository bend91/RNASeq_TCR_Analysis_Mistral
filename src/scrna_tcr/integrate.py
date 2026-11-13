"""Batch integration (scVI or PCA fallback), neighbours, UMAP, clustering.

Integration runs on the ``gex`` modality; embeddings are mirrored to the
MuData level so scirpy/plots can use them.
"""
from __future__ import annotations

import logging

import scanpy as sc
from mudata import MuData

log = logging.getLogger("scrna_tcr.integration")


def _hvg_pca(gex, cfg: dict) -> None:
    n_top = min(cfg["n_top_genes"], gex.n_vars)
    sc.pp.highly_variable_genes(gex, n_top_genes=n_top, flavor="seurat_v3", subset=False)
    sc.pp.pca(gex, n_comps=min(cfg["n_pcs"], gex.n_vars - 1), use_highly_variable=True)
    gex.obsm["X_embed"] = gex.obsm["X_pca"].copy()


def _scvi(gex, cfg: dict) -> None:
    import scvi

    scvi.model.SCVI.setup_anndata(gex, batch_key=cfg["batch_key"])
    model = scvi.model.SCVI(gex, n_latent=20, n_layers=2)
    model.train(max_epochs=cfg["scvi_max_epochs"], early_stopping=True, train_size=0.9)
    gex.obsm["X_embed"] = model.get_latent_representation().astype("float32")


def run_integration(mdata: MuData, cfg: dict) -> MuData:
    gex = mdata.mod["gex"]
    method = cfg["method"]
    if method == "scvi":
        try:
            _scvi(gex, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("scVI failed (%s) - falling back to PCA", exc)
            method = "pca"
    if method == "pca":
        _hvg_pca(gex, cfg)
    sc.pp.neighbors(gex, n_neighbors=cfg["n_neighbors"], use_rep="X_embed")
    sc.tl.umap(gex)
    mdata.obsm["X_embed"] = gex.obsm["X_embed"]
    mdata.obsm["X_umap"] = gex.obsm["X_umap"]
    log.info("integration done (%s)", method)
    return mdata
