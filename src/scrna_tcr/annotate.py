"""Cell-state annotation via gene-signature scoring."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc
from mudata import MuData

from .access import gex_x_gene, get_obs, sync_obs

log = logging.getLogger("scrna_tcr.annotate")


def score_signatures(mdata: MuData, signatures: dict[str, list[str]]) -> pd.DataFrame:
    """Score each signature with scanpy; genes missing from the panel are ignored."""
    gex = mdata.mod["gex"]
    scores = pd.DataFrame(index=mdata.obs_names)
    for name, genes in signatures.items():
        present = [g for g in genes if g in gex.var_names]
        if len(present) < 2:
            log.warning("signature '%s': only %d genes present - skipped", name, len(present))
            continue
        col = f"score_{name}"
        sc.tl.score_genes(gex, gene_list=present, score_name=col, use_raw=False)
        scores[name] = gex.obs[col].to_numpy()
    mdata.obsm["sig_scores"] = scores
    return scores


def assign_states(mdata: MuData, scores: pd.DataFrame, state_key: str = "cell_state") -> None:
    labels = scores.idxmax(axis=1)
    mdata.obs[state_key] = pd.Categorical(labels)


def restrict_to_t_cells(mdata: MuData, cd3_gene: str = "CD3D") -> MuData:
    """Keep cells with detected TCR chains or CD3D expression."""
    try:
        receptor_type = get_obs(mdata, "receptor_type")
        has_tcr = (receptor_type != "no IR").to_numpy()
    except KeyError:
        has_tcr = np.zeros(mdata.n_obs, dtype=bool)
    if cd3_gene in mdata.mod["gex"].var_names:
        has_tcr = has_tcr | (gex_x_gene(mdata, cd3_gene) > 0)
    n_before = mdata.n_obs
    mdata = mdata[has_tcr].copy()
    mdata.update()
    sync_obs(mdata)
    log.info("T-cell restriction: %d -> %d cells", n_before, mdata.n_obs)
    return mdata


def cluster(mdata: MuData, resolution: float) -> None:
    gex = mdata.mod["gex"]
    sc.tl.leiden(gex, resolution=resolution, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    sync_obs(mdata)
