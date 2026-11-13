"""Per-group TCR clonotype enrichment analysis."""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("scrna_tcr.enrichment")

EPS = 1e-6


def representative_cdr3(mdata) -> pd.Series:
    """Most common TRB (fallback TRA) CDR3 amino-acid sequence per clonotype."""
    import scirpy as ir

    if "airr" not in mdata.mod:
        return pd.Series(dtype=object, name="cdr3")
    obs = mdata.obs
    if "clone_id" not in obs.columns:
        return pd.Series(dtype=object, name="cdr3")
    chains = ir.get.airr(mdata, "junction_aa", chain=("VDJ_1", "VJ_1"))
    if isinstance(chains, pd.Series):
        chains = chains.to_frame()
    col = "IR_VDJ_1_junction_aa" if "IR_VDJ_1_junction_aa" in chains else chains.columns[0]
    cdr3 = chains[col].reindex(obs.index)
    sub = pd.DataFrame({"clone_id": obs["clone_id"].astype(str).values,
                        "cdr3": cdr3.values},
                       index=obs.index)
    sub = sub[sub["clone_id"].notna() & sub["cdr3"].notna()]
    if sub.empty:
        return pd.Series(dtype=object, name="cdr3")
    return sub.groupby("clone_id")["cdr3"].agg(lambda s: s.mode().iat[0]).rename("cdr3")


def clonotype_patient_frequency(adata: ad.AnnData, patient_key: str,
                                tissue_key: str) -> pd.DataFrame:
    """Frequency of each clonotype within each patient x tissue repertoire."""
    df = adata.obs[adata.obs["clone_id"].notna()].copy()
    denom = df.groupby([patient_key, tissue_key], observed=True).size().rename("n_cells_total")
    counts = (
        df.groupby([patient_key, tissue_key, "treatment", "clone_id"], observed=True)
        .size().rename("n_cells").reset_index()
    )
    counts = counts.merge(denom, on=[patient_key, tissue_key])
    counts["freq"] = counts["n_cells"] / counts["n_cells_total"]
    return counts


def enrich_group(
    freqs: pd.DataFrame,
    group: str,
    group_key: str,
    patient_key: str,
    tissue_key: str,
    min_clonotype_cells: int,
) -> pd.DataFrame:
    """One-vs-rest enrichment of each clonotype for ``group``."""
    in_grp = freqs[freqs[group_key] == group]
    out_grp = freqs[freqs[group_key] != group]

    total_cells = freqs.groupby("clone_id")["n_cells"].sum()
    tested = total_cells[total_cells >= min_clonotype_cells].index

    rows = []
    for clone in tested:
        g = in_grp[in_grp["clone_id"] == clone]["freq"].to_numpy()
        r = out_grp[out_grp["clone_id"] == clone]["freq"].to_numpy()
        if g.size == 0:
            continue
        # presence / absence at the patient-tissue level
        n_g = freqs[(freqs[group_key] == group) & (freqs["clone_id"] == clone)].shape[0]
        n_g_total = freqs[freqs[group_key] == group].drop_duplicates(
            [patient_key, tissue_key]).shape[0]
        n_r = freqs[(freqs[group_key] != group) & (freqs["clone_id"] == clone)].shape[0]
        n_r_total = freqs[freqs[group_key] != group].drop_duplicates(
            [patient_key, tissue_key]).shape[0]
        _, p_fisher = fisher_exact([[n_g, n_g_total - n_g], [n_r, n_r_total - n_r]])

        if g.size >= 2 and r.size >= 2:
            _, p_mwu = mannwhitneyu(g, r, alternative="two-sided")
        else:
            p_mwu = np.nan

        rows.append({
            "clone_id": clone,
            "mean_freq_in_group": float(g.mean()),
            "mean_freq_rest": float(r.mean()) if r.size else 0.0,
            "log2FC": float(np.log2((g.mean() + EPS) / (r.mean() + EPS))) if r.size else np.inf,
            "n_patients_in_group": int(
                freqs[(freqs[group_key] == group) & (freqs["clone_id"] == clone)][patient_key].nunique()),
            "p_fisher": p_fisher,
            "p_mwu": p_mwu,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_combined"] = res[["p_fisher", "p_mwu"]].min(axis=1)
    res["fdr"] = multipletests(res["p_combined"], method="fdr_bh")[1]
    # ranking: significance first, then replication across patients, then effect size
    res["log2FC_rank"] = res["log2FC"].replace(np.inf, 50.0)
    res = res.sort_values(
        ["fdr", "n_patients_in_group", "log2FC_rank"],
        ascending=[True, False, False],
    ).drop(columns="log2FC_rank").reset_index(drop=True)
    return res


def run_enrichment(adata: ad.AnnData, cfg: dict) -> dict[str, pd.DataFrame]:
    freqs = clonotype_patient_frequency(
        adata, cfg["patient_key"], cfg["tissue_key"])
    cdr3 = representative_cdr3(adata)

    results = {}
    for group in sorted(adata.obs[cfg["group_key"]].astype(str).unique()):
        res = enrich_group(
            freqs, group, cfg["group_key"], cfg["patient_key"],
            cfg["tissue_key"], cfg["min_clonotype_cells"],
        )
        if res.empty:
            log.warning("no clonotypes testable for group %s", group)
            continue
        res = res.merge(cdr3, left_on="clone_id", right_index=True, how="left")
        res["enriched_in"] = group
        res["significant"] = res["fdr"] < cfg["fdr_alpha"]
        results[group] = res
        log.info("group %s: %d clonotypes tested, %d significant (FDR<%.2f)",
                 group, len(res), int(res["significant"].sum()), cfg["fdr_alpha"])
    return results
