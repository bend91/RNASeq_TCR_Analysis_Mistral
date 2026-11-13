#!/usr/bin/env python3
"""Run the paired scRNA-seq + TCR-seq analysis pipeline.

Usage:
    python run_pipeline.py --config config.yaml --stage all
    python run_pipeline.py --config config.yaml --stage qc integrate tcr
Stages: qc integrate annotate tcr enrichment phenotype trafficking all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import anndata as ad
import mudata

from scrna_tcr import (annotate, config as cfgmod, enrichment, integrate,
                       io_utils, phenotype, plots, qc, tcr, trafficking)

log = logging.getLogger("scrna_tcr.pipeline")

STAGES = ["qc", "integrate", "annotate", "tcr", "enrichment", "phenotype", "trafficking"]


def _read(path: Path):
    return mudata.read(path)


def _write(mdata, path: Path) -> None:
    mdata.write(path)


def _plot_view(mdata) -> ad.AnnData:
    """Minimal AnnData carrying obs/obsm for scanpy plotting."""
    view = ad.AnnData(
        X=None,
        obs=mdata.obs.copy(),
        var=__import__("pandas").DataFrame(index=["dummy"]),
    )
    for k in ("X_umap", "X_embed"):
        if k in mdata.obsm:
            view.obsm[k] = mdata.obsm[k]
    return view


def stage_qc(cfg: dict, dirs: dict):
    gex, airr = io_utils.load_dataset(cfg)
    gex = qc.run_qc(gex, cfg["qc"])
    mdata = io_utils.merge_gex_with_ir(gex, airr)
    if cfg["annotation"].get("t_cell_only", True):
        mdata = annotate.restrict_to_t_cells(mdata, cfg["annotation"].get("cd3_gene", "CD3D"))
    _write(mdata, dirs["processed"] / "01_qc.h5mu")
    if cfg["output"].get("figures", True):
        plots.qc_violins(_plot_view(mdata), dirs["figures"])
    return mdata


def stage_integrate(cfg: dict, dirs: dict):
    mdata = _read(dirs["processed"] / "01_qc.h5mu")
    mdata = integrate.run_integration(mdata, cfg["integration"])
    annotate.cluster(mdata, cfg["annotation"]["cluster_res"])
    _write(mdata, dirs["processed"] / "02_integrated.h5mu")
    return mdata


def stage_annotate(cfg: dict, dirs: dict):
    mdata = _read(dirs["processed"] / "02_integrated.h5mu")
    scores = annotate.score_signatures(mdata, cfg["annotation"]["signatures"])
    annotate.assign_states(mdata, scores, cfg["phenotype"]["state_key"])
    _write(mdata, dirs["processed"] / "03_annotated.h5mu")
    if cfg["output"].get("figures", True):
        plots.umap_states(_plot_view(mdata), cfg["phenotype"]["state_key"], dirs["figures"])
    return mdata


def stage_tcr(cfg: dict, dirs: dict):
    mdata = _read(dirs["processed"] / "03_annotated.h5mu")
    mdata = tcr.run_tcr_analysis(mdata, cfg["tcr"])
    _write(mdata, dirs["processed"] / "04_tcells_tcr.h5mu")
    div = tcr.diversity_table(mdata, cfg["enrichment"]["patient_key"],
                              cfg["enrichment"]["tissue_key"])
    div.to_csv(dirs["tables"] / "tcr_diversity.csv", index=False)
    if cfg["output"].get("figures", True):
        view = _plot_view(mdata)
        plots.umap_expansion(view, dirs["figures"])
        plots.top_clonotype_bars(view, cfg["enrichment"]["group_key"], dirs["figures"])
        plots.diversity_plot(div, cfg["enrichment"]["patient_key"], dirs["figures"])
    return mdata


def stage_enrichment(cfg: dict, dirs: dict):
    import pandas as pd

    mdata = _read(dirs["processed"] / "04_tcells_tcr.h5mu")
    results = enrichment.run_enrichment(mdata, cfg["enrichment"])
    for group, res in results.items():
        safe = str(group).replace("/", "_")
        res.to_csv(dirs["tables"] / f"tcr_enrichment_{safe}.csv", index=False)
        if cfg["output"].get("figures", True):
            plots.enrichment_volcano(res, safe, dirs["figures"])
    return results


def stage_phenotype(cfg: dict, dirs: dict, enrichment_results=None):
    import pandas as pd

    mdata = _read(dirs["processed"] / "04_tcells_tcr.h5mu")
    if enrichment_results is None:
        enrichment_results = {}
        for group in sorted(mdata.obs[cfg["enrichment"]["group_key"]].astype(str).unique()):
            safe = str(group).replace("/", "_")
            path = dirs["tables"] / f"tcr_enrichment_{safe}.csv"
            if path.exists():
                enrichment_results[group] = pd.read_csv(path)
    pheno, sig = phenotype.run_phenotyping(mdata, enrichment_results, cfg["phenotype"])
    pheno.to_csv(dirs["tables"] / "enriched_tcr_phenotypes.csv", index=False)
    sig.to_csv(dirs["tables"] / "enriched_tcr_signature_tests.csv", index=False)
    if cfg["output"].get("figures", True):
        plots.phenotype_heatmap(pheno, dirs["figures"])
        plots.state_composition_bars(pheno, dirs["figures"])
    return pheno


def stage_trafficking(cfg: dict, dirs: dict):
    mdata = _read(dirs["processed"] / "04_tcells_tcr.h5mu")
    res = trafficking.run_trafficking(mdata, cfg["trafficking"])
    res["overlap_per_patient"].to_csv(dirs["tables"] / "ln_tumor_overlap_per_patient.csv",
                                      index=False)
    res["shared_clonotypes"].to_csv(dirs["tables"] / "shared_clonotypes_ln_tumor.csv",
                                    index=False)
    res["overlap_group_comparison"].to_csv(dirs["tables"] / "overlap_group_comparison.csv",
                                           index=False)
    res["phenotype_shifts"].to_csv(dirs["tables"] / "shared_clone_phenotype_shifts.csv",
                                   index=False)
    if cfg["output"].get("figures", True):
        plots.overlap_boxes(res["overlap_per_patient"], dirs["figures"])
        plots.shift_heatmap(res["shared_clonotypes"], dirs["figures"])
        plots.transition_heatmap(res["state_transitions"], dirs["figures"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", nargs="+", default=["all"], choices=STAGES + ["all"])
    args = ap.parse_args()

    cfgmod.setup_logging()
    cfg = cfgmod.load_config(args.config)
    dirs = cfgmod.results_dirs(cfg)
    stages = STAGES if "all" in args.stage else args.stage

    enrichment_results = None
    for stage in stages:
        log.info("=== stage: %s ===", stage)
        if stage == "qc":
            stage_qc(cfg, dirs)
        elif stage == "integrate":
            stage_integrate(cfg, dirs)
        elif stage == "annotate":
            stage_annotate(cfg, dirs)
        elif stage == "tcr":
            stage_tcr(cfg, dirs)
        elif stage == "enrichment":
            enrichment_results = stage_enrichment(cfg, dirs)
        elif stage == "phenotype":
            stage_phenotype(cfg, dirs, enrichment_results)
        elif stage == "trafficking":
            stage_trafficking(cfg, dirs)
    log.info("pipeline finished - results in %s", dirs["base"])


if __name__ == "__main__":
    main()
