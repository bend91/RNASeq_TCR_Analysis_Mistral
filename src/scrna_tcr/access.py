"""Helpers for working with MuData (gex + airr modalities)."""
from __future__ import annotations

import anndata as ad
import mudata
import numpy as np
import pandas as pd
from mudata import MuData

GEX_MOD = "gex"
AIRR_MOD = "airr"


def make_mdata(gex: ad.AnnData, airr: ad.AnnData | None) -> MuData:
    mods = {GEX_MOD: gex}
    if airr is not None:
        mods[AIRR_MOD] = airr
    mdata = MuData(mods)
    mdata.update()
    sync_obs(mdata)
    return mdata


def sync_obs(mdata: MuData) -> None:
    """Mirror mod-level obs columns into mdata.obs without prefixes.

    Global (unprefixed) columns are what the analysis modules use; scirpy
    writes results to the airr mod where MuData exposes them as
    ``airr:<column>``.
    """
    for mod_name in (GEX_MOD, AIRR_MOD):
        if mod_name not in mdata.mod:
            continue
        for col in mdata.mod[mod_name].obs.columns:
            if col not in mdata.obs.columns:
                mdata.obs[col] = mdata.mod[mod_name].obs[col]


def get_obs(mdata: MuData, col: str) -> pd.Series:
    for name in (col, f"{AIRR_MOD}:{col}", f"{GEX_MOD}:{col}"):
        if name in mdata.obs.columns:
            return mdata.obs[name]
    raise KeyError(f"column {col!r} not found in mdata.obs")


def gex_x_gene(mdata: MuData, gene: str) -> np.ndarray:
    gex = mdata.mod[GEX_MOD]
    x = gex[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).ravel()


def read_mdata(path) -> MuData:
    return mudata.read(path)


def write_mdata(mdata: MuData, path) -> None:
    mdata.write(path)
