"""Figure generation for all pipeline stages."""
from __future__ import annotations

import logging

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
log = logging.getLogger("scrna_tcr.plots")


def qc_violins(adata: ad.AnnData, fig_dir) -> None:
    metrics = ["n_genes_by_counts", "total_counts"]
    if "pct_counts_mt" in adata.obs:
        metrics.append("pct_counts_mt")
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sns.violinplot(data=adata.obs, x="sample_id", y=m, ax=ax, cut=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_qc_violins.png", dpi=150)
    plt.close(fig)


def umap_states(adata: ad.AnnData, state_key: str, fig_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color=state_key, ax=axes[0], show=False, frameon=False)
    sc.pl.umap(adata, color="treatment", ax=axes[1], show=False, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_umap_states.png", dpi=150)
    plt.close(fig)


def umap_expansion(adata: ad.AnnData, fig_dir) -> None:
    if "expansion" not in adata.obs:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(adata, color="expansion", ax=ax, show=False, frameon=False,
               title="Clonal expansion")
    fig.savefig(fig_dir / "03_umap_expansion.png", dpi=150)
    plt.close(fig)


def top_clonotype_bars(adata: ad.AnnData, group_key: str, fig_dir, top_n: int = 10) -> None:
    if "clone_id" not in adata.obs:
        return
    df = adata.obs[adata.obs["clone_id"].notna()]
    ct = pd.crosstab(df[group_key], df["clone_id"], normalize="index")
    top = df["clone_id"].value_counts().head(top_n).index
    ct[top].plot(kind="bar", stacked=True, figsize=(8, 5), colormap="tab20")
    plt.ylabel("fraction of T cells")
    plt.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "04_top_clonotypes_by_group.png", dpi=150)
    plt.close()


def diversity_plot(div: pd.DataFrame, patient_key: str, fig_dir) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=div, x=patient_key, y="shannon", hue="tissue", ax=ax)
    ax.set_title("TCR Shannon diversity")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_diversity.png", dpi=150)
    plt.close(fig)


def enrichment_volcano(res: pd.DataFrame, group: str, fig_dir) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = res["log2FC"].replace(np.inf, res["log2FC"].replace([np.inf], np.nan).max() * 1.1)
    y = -np.log10(res["fdr"].clip(lower=1e-300))
    ax.scatter(x, y, s=8, c=res["significant"].map({True: "crimson", False: "grey"}))
    ax.set_xlabel("log2FC clonotype frequency (group vs rest)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title(f"Clonotype enrichment: {group}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"06_enrichment_{group}.png", dpi=150)
    plt.close(fig)


def phenotype_heatmap(pheno: pd.DataFrame, fig_dir) -> None:
    if pheno.empty:
        return
    sig_cols = [c for c in pheno.columns if c.startswith("log2FC_") and "enrichment" not in c]
    if not sig_cols:
        return
    mat = pheno.set_index(["group", "clone_id"])[sig_cols]
    mat.columns = [c[len("log2FC_"):] for c in sig_cols]
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * mat.shape[1], 0.35 * mat.shape[0] + 2))
    sns.heatmap(mat, center=0, cmap="vlag", ax=ax, cbar_kws={"label": "log2FC vs background"})
    ax.set_title("Signature scores of enriched clonotypes")
    fig.tight_layout()
    fig.savefig(fig_dir / "07_enriched_clonotype_phenotypes.png", dpi=150)
    plt.close(fig)


def state_composition_bars(pheno: pd.DataFrame, fig_dir) -> None:
    if pheno.empty:
        return
    frac_cols = [c for c in pheno.columns if c.startswith("frac_")
                 and not c.startswith("frac_tissue")]
    if not frac_cols:
        return
    comp = pheno.set_index(["group", "clone_id"])[frac_cols]
    comp.columns = [c[len("frac_"):] for c in frac_cols]
    comp.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
    plt.ylabel("fraction of clonotype cells")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "08_enriched_clonotype_states.png", dpi=150)
    plt.close()


def overlap_boxes(overlap: pd.DataFrame, fig_dir) -> None:
    metrics = ["n_shared", "jaccard", "morisita_horn", "trafficking_index"]
    metrics = [m for m in metrics if m in overlap.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sns.boxplot(data=overlap, x="treatment", y=m, ax=ax)
        sns.stripplot(data=overlap, x="treatment", y=m, ax=ax, color="k", size=4)
        ax.set_title(m)
    fig.tight_layout()
    fig.savefig(fig_dir / "09_ln_tumor_overlap_by_treatment.png", dpi=150)
    plt.close(fig)


def shift_heatmap(shared: pd.DataFrame, fig_dir) -> None:
    shift_cols = [c for c in shared.columns if c.startswith("shift_")]
    if not shift_cols or shared.empty:
        return
    mat = shared.groupby("treatment", observed=True)[shift_cols].mean()
    mat.columns = [c[len("shift_"):] for c in shift_cols]
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * mat.shape[1], 0.6 * mat.shape[0] + 1.5))
    sns.heatmap(mat, center=0, cmap="vlag", ax=ax,
                cbar_kws={"label": "mean signature shift LN -> tumour"})
    ax.set_title("Shared-clonotype phenotype shift (LN -> tumour)")
    fig.tight_layout()
    fig.savefig(fig_dir / "10_shared_clone_phenotype_shift.png", dpi=150)
    plt.close(fig)


def transition_heatmap(transitions: pd.DataFrame, fig_dir) -> None:
    if transitions is None or transitions.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(transitions, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("state in tumour")
    ax.set_ylabel("state in lymph node")
    ax.set_title("Shared clonotypes: state transition LN -> tumour")
    fig.tight_layout()
    fig.savefig(fig_dir / "11_state_transitions.png", dpi=150)
    plt.close(fig)
