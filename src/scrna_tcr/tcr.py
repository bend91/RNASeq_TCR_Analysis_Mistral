"""TCR clonotype analysis with scirpy (MuData workflow)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scirpy as ir
from mudata import MuData

from .access import sync_obs

log = logging.getLogger("scrna_tcr.tcr")


def run_tcr_analysis(mdata: MuData, tcr_cfg: dict) -> MuData:
    if "airr" not in mdata.mod:
        log.warning("no TCR data - skipping TCR analysis")
        return mdata

    ir.tl.chain_qc(mdata)

    if tcr_cfg["clonotype_metric"] == "identity":
        ir.pp.ir_dist(mdata, metric="identity", sequence="aa",
                      key_added="ir_dist_aa_identity")
        ir.tl.define_clonotype_clusters(
            mdata, sequence="aa", metric="identity",
            receptor_arms="all", dual_ir="any",
            same_v_gene=False, within_group=None,
            distance_key="ir_dist_aa_identity", key_added="clone_id",
        )
    else:
        ir.pp.ir_dist(mdata, metric=tcr_cfg["clonotype_metric"], sequence="aa",
                      key_added=tcr_cfg["clonotype_distance_key"])
        ir.tl.define_clonotype_clusters(
            mdata, sequence="aa", metric=tcr_cfg["clonotype_metric"],
            receptor_arms="all", dual_ir="any",
            same_v_gene=False, within_group=None,
            distance_key=tcr_cfg["clonotype_distance_key"], key_added="clone_id",
        )

    ir.tl.clonal_expansion(mdata, breakpoints=tcr_cfg["expansion_bins"],
                           key_added="expansion")
    sync_obs(mdata)
    log.info("defined %d clonotypes", mdata.obs["clone_id"].nunique())
    return mdata


def clonotype_frequency_table(mdata: MuData) -> pd.DataFrame:
    """Per-sample clonotype counts and frequencies (long format)."""
    obs = mdata.obs
    df = obs[obs["clone_id"].notna()].copy()
    sample_sizes = df.groupby("sample_id", observed=True).size().rename("n_cells_sample")
    counts = (
        df.groupby(["sample_id", "patient", "treatment", "tissue", "clone_id"], observed=True)
        .size().rename("n_cells").reset_index()
    )
    counts = counts.merge(sample_sizes, on="sample_id")
    counts["freq"] = counts["n_cells"] / counts["n_cells_sample"]
    return counts


def diversity_table(mdata: MuData, patient_key: str = "patient",
                    tissue_key: str = "tissue") -> pd.DataFrame:
    """Shannon diversity per patient x tissue."""
    rows = []
    obs = mdata.obs
    df = obs[obs["clone_id"].notna()]
    for (pat, tis), sub in df.groupby([patient_key, tissue_key], observed=True):
        counts = sub["clone_id"].value_counts().to_numpy()
        p = counts / counts.sum()
        shannon = float(-(p * np.log(p)).sum())
        rows.append({patient_key: pat, tissue_key: tis,
                     "n_clonotypes": len(counts), "shannon": shannon})
    return pd.DataFrame(rows)
