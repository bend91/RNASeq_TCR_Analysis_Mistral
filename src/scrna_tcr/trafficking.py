"""Lymph-node -> tumour shared-clonotype (trafficking) analysis."""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

log = logging.getLogger("scrna_tcr.trafficking")

EPS = 1e-6


def morisita_horn(a: np.ndarray, b: np.ndarray) -> float:
    """Morisita-Horn overlap of two count vectors (aligned on the union)."""
    sa, sb = a.sum(), b.sum()
    if sa == 0 or sb == 0:
        return np.nan
    num = 2.0 * float(np.sum(a * b))
    den = (np.sum(a**2) / sa**2 + np.sum(b**2) / sb**2) * sa * sb
    return num / den if den > 0 else np.nan


def clone_tissue_frequencies(adata: ad.AnnData, patient_key: str,
                             tissue_key: str) -> pd.DataFrame:
    """Wide table: rows = (patient, clone), cols = freq per tissue + counts."""
    df = adata.obs[adata.obs["clone_id"].notna()].copy()
    counts = df.groupby([patient_key, "treatment", tissue_key, "clone_id"],
                        observed=True).size().rename("n_cells").reset_index()
    denom = df.groupby([patient_key, tissue_key], observed=True).size().rename("n_total")
    counts = counts.merge(denom, on=[patient_key, tissue_key])
    counts["freq"] = counts["n_cells"] / counts["n_total"]
    return counts


def per_patient_overlap(counts: pd.DataFrame, patient_key: str,
                        ln_label: str, tumor_label: str) -> pd.DataFrame:
    rows = []
    for pat, sub in counts.groupby(patient_key, observed=True):
        ln = sub[sub[tissue_key_col(counts)] == ln_label]
        tu = sub[sub[tissue_key_col(counts)] == tumor_label]
        ln_clones = set(ln["clone_id"])
        tu_clones = set(tu["clone_id"])
        union = ln_clones | tu_clones
        ln_vec = (ln.set_index("clone_id")["n_cells"]
                  .reindex(union, fill_value=0).to_numpy())
        tu_vec = (tu.set_index("clone_id")["n_cells"]
                  .reindex(union, fill_value=0).to_numpy())
        tu_total = tu["n_cells"].sum()
        shared_in_tumor = tu[tu["clone_id"].isin(ln_clones)]["n_cells"].sum()
        rows.append({
            patient_key: pat,
            "treatment": sub["treatment"].iat[0],
            "n_clones_ln": len(ln_clones),
            "n_clones_tumor": len(tu_clones),
            "n_shared": len(ln_clones & tu_clones),
            "jaccard": len(ln_clones & tu_clones) / len(union) if union else np.nan,
            "morisita_horn": morisita_horn(ln_vec, tu_vec),
            "trafficking_index": float(shared_in_tumor / tu_total) if tu_total else np.nan,
        })
    return pd.DataFrame(rows)


def tissue_key_col(counts: pd.DataFrame) -> str:
    for cand in ("tissue", "tissue_key"):
        if cand in counts.columns:
            return cand
    raise KeyError("tissue column not found")


def shared_clonotype_table(adata: ad.AnnData, counts: pd.DataFrame,
                           patient_key: str, ln_label: str, tumor_label: str,
                           sig_scores: pd.DataFrame, state_key: str,
                           expansion_fold: float) -> pd.DataFrame:
    tis = tissue_key_col(counts)
    # dominant cell state + mean signature scores per clone x tissue
    obs = adata.obs[adata.obs["clone_id"].notna()].copy()
    obs["clone_id"] = obs["clone_id"].astype(str)
    for sig in sig_scores.columns:
        obs[f"sig_{sig}"] = sig_scores.loc[obs.index, sig].to_numpy()

    def agg_tissue(sub: pd.DataFrame) -> pd.Series:
        out = {"n_cells": len(sub)}
        vc = sub[state_key].value_counts()
        out["dominant_state"] = vc.idxmax()
        out["dominant_state_frac"] = vc.iloc[0] / len(sub)
        for sig in sig_scores.columns:
            out[f"mean_{sig}"] = float(sub[f"sig_{sig}"].mean())
        return out

    agg = (obs.groupby([patient_key, tis, "clone_id"], observed=True)
           .apply(agg_tissue, include_groups=False))
    wide = agg.unstack(tis)

    rows = []
    for (pat, clone), _ in counts.groupby([patient_key, "clone_id"], observed=True):
        sub = counts[(counts[patient_key] == pat) & (counts["clone_id"] == clone)]
        tissues = set(sub[tis])
        if not {ln_label, tumor_label} <= tissues:
            continue
        ln_c = int(sub[sub[tis] == ln_label]["n_cells"].iat[0])
        tu_c = int(sub[sub[tis] == tumor_label]["n_cells"].iat[0])
        ln_f = float(sub[sub[tis] == ln_label]["freq"].iat[0])
        tu_f = float(sub[sub[tis] == tumor_label]["freq"].iat[0])
        row = {
            patient_key: pat,
            "treatment": sub["treatment"].iat[0],
            "clone_id": clone,
            "n_cells_ln": ln_c, "n_cells_tumor": tu_c,
            "freq_ln": ln_f, "freq_tumor": tu_f,
            "log2FC_tumor_vs_ln": np.log2((tu_f + EPS) / (ln_f + EPS)),
            "expanded_in_tumor": tu_f > ln_f * expansion_fold,
        }
        ln_key, tu_key = (pat, ln_label, clone), (pat, tumor_label, clone)
        if ln_key in agg.index and tu_key in agg.index:
            ln_stats = agg.loc[ln_key]
            tu_stats = agg.loc[tu_key]
            row["dominant_state_ln"] = ln_stats["dominant_state"]
            row["dominant_state_tumor"] = tu_stats["dominant_state"]
            for sig in sig_scores.columns:
                row[f"shift_{sig}"] = float(tu_stats[f"mean_{sig}"] - ln_stats[f"mean_{sig}"])
        rows.append(row)
    return pd.DataFrame(rows)


def group_comparison(overlap: pd.DataFrame, patient_key: str,
                     group_key: str = "treatment") -> pd.DataFrame:
    """Test overlap metrics across treatment groups."""
    metrics = ["n_shared", "jaccard", "morisita_horn", "trafficking_index"]
    rows = []
    for metric in metrics:
        groups = [g[metric].dropna().to_numpy()
                  for _, g in overlap.groupby(group_key, observed=True)]
        names = [str(n) for n, _ in overlap.groupby(group_key, observed=True)]
        if len(groups) >= 3 and all(g.size >= 2 for g in groups):
            p_kw = kruskal(*groups).pvalue
        else:
            p_kw = np.nan
        for name, g in zip(names, groups):
            rows.append({"metric": metric, "group": name, "n_patients": g.size,
                         "mean": float(np.mean(g)) if g.size else np.nan,
                         "median": float(np.median(g)) if g.size else np.nan,
                         "p_kruskal_all_groups": p_kw})
    return pd.DataFrame(rows)


def phenotype_shift_test(shared: pd.DataFrame) -> pd.DataFrame:
    """Paired test (LN -> tumour) of signature shifts across shared clonotypes, per group."""
    rows = []
    sig_cols = [c for c in shared.columns if c.startswith("shift_")]
    for group, sub in shared.groupby("treatment", observed=True):
        for col in sig_cols:
            sig = col[len("shift_"):]
            vals = sub[col].dropna().to_numpy()
            if vals.size < 3 or np.allclose(vals, 0):
                p = np.nan
            else:
                try:
                    p = wilcoxon(vals).pvalue
                except ValueError:
                    p = np.nan
            rows.append({
                "treatment": group, "signature": sig,
                "n_shared_clones": vals.size,
                "mean_shift": float(np.mean(vals)) if vals.size else np.nan,
                "median_shift": float(np.median(vals)) if vals.size else np.nan,
                "p_wilcoxon": p,
                "fraction_increased": float((vals > 0).mean()) if vals.size else np.nan,
            })
    return pd.DataFrame(rows)


def state_transition_matrix(shared: pd.DataFrame, patient_key: str) -> pd.DataFrame:
    """Dominant-state transitions LN -> tumour across shared clonotypes."""
    sub = shared.dropna(subset=["dominant_state_ln", "dominant_state_tumor"])
    if sub.empty:
        return pd.DataFrame()
    return pd.crosstab(sub["dominant_state_ln"], sub["dominant_state_tumor"])


def run_trafficking(adata: ad.AnnData, cfg: dict) -> dict:
    patient_key = cfg.get("patient_key", "patient")
    ln_label = cfg["ln_label"]
    tumor_label = cfg["tumor_label"]

    counts = clone_tissue_frequencies(adata, patient_key, "tissue")
    overlap = per_patient_overlap(counts, patient_key, ln_label, tumor_label)
    shared = shared_clonotype_table(
        adata, counts, patient_key, ln_label, tumor_label,
        adata.obsm["sig_scores"], cfg.get("state_key", "cell_state"),
        cfg["expansion_fold"],
    )
    overlap_cmp = group_comparison(overlap, patient_key)
    shift_test = phenotype_shift_test(shared)
    transitions = state_transition_matrix(shared, patient_key)

    log.info("trafficking: %d shared clonotypes across %d patients",
             len(shared), overlap[patient_key].nunique())
    return {
        "overlap_per_patient": overlap,
        "shared_clonotypes": shared,
        "overlap_group_comparison": overlap_cmp,
        "phenotype_shifts": shift_test,
        "state_transitions": transitions,
    }
