#!/usr/bin/env python3
"""Generate a small synthetic paired scRNA-seq + TCR-seq dataset for smoke tests.

Layout: 3 treatments x 2 patients x 2 tissues (LN, tumour) = 12 samples.
Clonotypes: patient-shared LN<->tumour clones, tissue-private clones, and
treatment-specific public tumour clones (so enrichment hits exist by design).
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RNG = np.random.default_rng(42)

MARKERS = {
    "Naive": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "ACTN1"],
    "Central_mem": ["CCR7", "SELL", "IL7R", "GPR183"],
    "Effector_mem": ["GZMK", "GZMA", "CCL5", "CCL4", "EOMES", "CXCR3"],
    "Cytotoxic": ["GZMB", "PRF1", "NKG7", "GNLY", "FGFBP2", "KLRD1", "SPON2"],
    "Exhausted": ["PDCD1", "TIGIT", "LAG3", "HAVCR2", "TOX", "LAYN", "CXCL13", "ENTPD1"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"],
    "Proliferating": ["MKI67", "TOP2A", "STMN1", "TYMS"],
    "Tfh": ["CXCR5", "BCL6", "PDCD1", "CD200", "IL21"],
}
ALL_MARKERS = sorted({g for v in MARKERS.values() for g in v})
N_RANDOM_GENES = 1200
GENES = [f"GENE{i:04d}" for i in range(N_RANDOM_GENES)] + ALL_MARKERS + ["CD3D"]

TREATMENTS = ["control", "antiPD1", "combo"]
PATIENTS = {"control": ["P1", "P2"], "antiPD1": ["P3", "P4"], "combo": ["P5", "P6"]}
TISSUES = ["LN", "tumor"]

STATE_BASE_FREQ = {
    "Naive": 0.25, "Central_mem": 0.25, "Effector_mem": 0.20,
    "Cytotoxic": 0.12, "Exhausted": 0.08, "Treg": 0.08, "Proliferating": 0.02,
}


def aa_seq(rng, n=14):
    aas = "ARNDCQEGHILKMFPSTWYV"
    return "C" + "".join(rng.choice(list(aas), n - 2)) + "F"


def nt_seq(rng, n=42):
    return "".join(rng.choice(list("ACGT"), n))


def _gzip_write(path: Path, text: str):
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def write_10x_mtx(out_dir: Path, counts: sp.csr_matrix, barcodes: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    from scipy.io import mmwrite
    mmwrite(out_dir / "matrix.mtx", counts.T)  # 10x stores genes x cells
    with open(out_dir / "matrix.mtx", "rb") as f_in, gzip.open(out_dir / "matrix.mtx.gz", "wb") as f_out:
        f_out.writelines(f_in)
    (out_dir / "matrix.mtx").unlink()
    _gzip_write(out_dir / "barcodes.tsv.gz", "\n".join(barcodes) + "\n")
    _gzip_write(out_dir / "features.tsv.gz",
                "".join(f"{g}\t{g}\tGene Expression\n" for g in GENES))


def simulate_gex(n_cells: int, states: list[str]) -> sp.csr_matrix:
    X = RNG.poisson(0.5, size=(n_cells, len(GENES))).astype(np.float32)
    gene_idx = {g: i for i, g in enumerate(GENES)}
    states = np.array(states)
    for state in set(states):
        idx = np.where(states == state)[0]
        for g in MARKERS.get(state, []):
            X[idx, gene_idx[g]] += RNG.poisson(6, size=len(idx))
    X[:, gene_idx["CD3D"]] += RNG.poisson(4, size=n_cells)
    return sp.csr_matrix(X)


def make_airr_rows(cell_ids, clone_ids, rng) -> pd.DataFrame:
    rows = []
    for cid, clone in zip(cell_ids, clone_ids):
        tra, trb = clone
        for locus, cdr in (("TRA", tra), ("TRB", trb)):
            rows.append({
                "cell_id": cid,
                "locus": locus,
                "junction_aa": cdr,
                "junction": nt_seq(rng),
                "v_call": f"TR{locus[2]}V1-1",
                "j_call": f"TR{locus[2]}J1-1",
                "productive": True,
                "is_cell": True,
                "consensus_count": int(rng.integers(2, 20)),
                "duplicate_count": int(rng.integers(1, 10)),
            })
    return pd.DataFrame(rows)


def main(out_root: str = "data") -> Path:
    out_root = Path(out_root)
    meta_rows = []
    clone_counter = 0

    treatment_public_clones = {
        t: [f"pub_{t}_{k}" for k in range(4)] for t in TREATMENTS
    }
    clone_seqs = {}

    for treatment in TREATMENTS:
        for patient in PATIENTS[treatment]:
            n_traffic = 15
            traffic_ids = {}
            for k in range(n_traffic):
                clone_counter += 1
                cid = f"traffic_{patient}_{k}"
                traffic_ids[cid] = (aa_seq(RNG), aa_seq(RNG))
                clone_seqs[cid] = traffic_ids[cid]
            for cid in treatment_public_clones[treatment]:
                if cid not in clone_seqs:
                    clone_seqs[cid] = (aa_seq(RNG), aa_seq(RNG))

            for tissue in TISSUES:
                sample_id = f"{patient}_{tissue}"
                n_cells = 400
                if tissue == "LN":
                    probs = np.array(list(STATE_BASE_FREQ.values()))
                    probs[4] *= 0.4  # less exhausted in LN
                else:
                    probs = np.array(list(STATE_BASE_FREQ.values()))
                    probs[4] *= 2.0  # more exhausted in tumour
                probs /= probs.sum()
                states = RNG.choice(list(STATE_BASE_FREQ), size=n_cells, p=probs).tolist()

                # ---- assign clonotypes --------------------------------------
                clone_assign = []
                pool = []

                for cid, seqs in traffic_ids.items():
                    # trafficking clones: present in both tissues for this patient
                    n_ln = RNG.poisson(4)
                    n_tu = RNG.poisson(6 if treatment == "combo" else 2)
                    pool += [(cid, seqs)] * n_ln + [(cid, seqs)] * n_tu

                if tissue == "tumor":
                    for cid in treatment_public_clones[treatment]:
                        seqs = clone_seqs[cid]
                        pool += [(cid, seqs)] * RNG.poisson(10 if treatment != "control" else 2)

                n_private = 30
                for k in range(n_private):
                    clone_counter += 1
                    cid = f"{sample_id}_priv{k}"
                    seqs = (aa_seq(RNG), aa_seq(RNG))
                    clone_seqs[cid] = seqs
                    pool += [(cid, seqs)] * RNG.poisson(3)

                pool = pool[:n_cells] + [(None, None)] * max(0, n_cells - len(pool))
                RNG.shuffle(pool)
                for i in range(n_cells):
                    cid, _ = pool[i]
                    clone_assign.append(cid)

                # naive cells are mostly non-clonal / private
                for i, state in enumerate(states):
                    if state in ("Naive",) and clone_assign[i] and not str(clone_assign[i]).startswith(f"{sample_id}_priv"):
                        clone_assign[i] = None

                # ---- write GEX ----------------------------------------------
                counts = simulate_gex(n_cells, states)
                barcodes = [f"{sample_id}_{i:08d}" for i in range(n_cells)]
                write_10x_mtx(out_root / "scrna" / sample_id, counts, barcodes)

                # ---- write TCR (AIRR) ---------------------------------------
                tcr_cells = [b for b, c in zip(barcodes, clone_assign) if c is not None]
                tcr_clones = [c for c in clone_assign if c is not None]
                airr = make_airr_rows(tcr_cells, [clone_seqs[c] for c in tcr_clones], RNG)
                airr.to_csv(out_root / "tcr" / f"{sample_id}.tsv", sep="\t", index=False)

                meta_rows.append({"sample_id": sample_id, "patient": patient,
                                  "treatment": treatment, "tissue": tissue})
                print(f"{sample_id}: {n_cells} cells, {len(tcr_cells)} with TCR")

    pd.DataFrame(meta_rows).to_csv(out_root / "sample_metadata.csv", index=False)
    print(f"wrote {len(meta_rows)} samples to {out_root}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(root)
