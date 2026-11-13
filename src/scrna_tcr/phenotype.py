"""Phenotyping of the most enriched clonotypes per group."""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("scrna_tcr.phenotype")


def _state_composition(cells: pd.DataFrame, state_key: str, all_states) -> pd.Series:
    vc = cells[state_key].value_counts(normalize=True)
    return vc.reindex(all_states, fill_value=0.0)


def phenotype_clonotype(
    adata: ad.AnnData,
    clone_id: str,
    sig_scores: pd.DataFrame,
    state_key: str,
    all_states,
) -> dict:
    is_clone = (adata.obs["clone_id"] == clone_id).to_numpy()
    cells = adata.obs.loc[is_clone]
    background = adata.obs.loc[~is_clone]

    out: dict = {
        "n_cells": int(is_clone.sum()),
        "dominant_state": cells[state_key].value_counts().idxmax(),
        **{f"frac_{s}": v for s, v in
           _state_composition(cells, state_key, all_states).items()},
    }

    # signature scores in clonotype vs background (Mann-Whitney + log2 mean ratio)
    for sig in sig_scores.columns:
        x = sig_scores.loc[is_clone, sig].to_numpy()
        y = sig_scores.loc[~is_clone, sig].to_numpy()
        p = np.nan
        if x.size >= 3 and y.size >= 3:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        out[f"mean_{sig}"] = float(np.nanmean(x)) if x.size else np.nan
        out[f"mean_{sig}_background"] = float(np.nanmean(y)) if y.size else np.nan
        out[f"log2FC_{sig}"] = float(
            np.log2((np.nanmean(x) + 1e-3) / (np.nanmean(y) + 1e-3)))
        out[f"p_{sig}"] = p

    # tissue distribution of the clonotype's cells
    tis = cells["tissue"].value_counts(normalize=True)
    for t in tis.index:
        out[f"frac_tissue_{t}"] = float(tis[t])
    return out


def run_phenotyping(adata: ad.AnnData, enrichment: dict[str, pd.DataFrame],
                    cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig_scores = adata.obsm[cfg["signature_key"]]
    state_key = cfg["state_key"]
    all_states = list(adata.obs[state_key].cat.categories) if isinstance(
        adata.obs[state_key].dtype, pd.CategoricalDtype) else sorted(adata.obs[state_key].unique())

    rows, sig_rows = [], []
    for group, res in enrichment.items():
        top = res.head(cfg.get("top_n", 20))
        log.info("phenotyping top %d clonotypes of group %s", len(top), group)
        for _, row in top.iterrows():
            if row["clone_id"] not in adata.obs["clone_id"].values:
                continue
            ph = phenotype_clonotype(adata, row["clone_id"], sig_scores, state_key, all_states)
            ph.update({
                "group": group,
                "clone_id": row["clone_id"],
                "cdr3": row.get("cdr3"),
                "log2FC_enrichment": row["log2FC"],
                "fdr_enrichment": row["fdr"],
            })
            rows.append(ph)
            for sig in sig_scores.columns:
                sig_rows.append({
                    "group": group, "clone_id": row["clone_id"], "cdr3": row.get("cdr3"),
                    "signature": sig,
                    "log2FC": ph[f"log2FC_{sig}"], "p": ph[f"p_{sig}"],
                })

    pheno = pd.DataFrame(rows)
    sig = pd.DataFrame(sig_rows)
    if not sig.empty:
        mask = sig["p"].notna()
        sig.loc[mask, "fdr"] = multipletests(sig.loc[mask, "p"], method="fdr_bh")[1]
        sig["fdr"] = sig["fdr"].fillna(1.0)
    return pheno, sig
