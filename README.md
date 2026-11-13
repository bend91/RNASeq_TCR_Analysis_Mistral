# Paired scRNA-seq + TCR-seq analysis pipeline

Modular Python pipeline (scanpy + scirpy + scvi-tools) for tumour/lymph-node
single-cell RNA-seq with paired TCR sequencing:

1. **QC** — per-sample 10x loading, cell/gene filtering, mito %, optional
   scrublet doublet removal, TCR chain QC, T-cell restriction.
2. **Integration** — PCA or scVI (batch = sample), neighbours, UMAP, Leiden.
3. **Annotation** — configurable gene-signature scoring (naive, memory,
   cytotoxic, exhausted, Treg, proliferating, Tfh) → per-cell state labels.
4. **TCR analysis** — global clonotype definition across all samples (required
   for tissue-tracking), clonal expansion, Shannon diversity per patient/tissue.
5. **Enrichment** — per treatment group, clonotypes enriched vs all other
   groups (Mann-Whitney U on patient-level frequencies + Fisher's exact test on
   presence/absence, BH-corrected), ranked with replication across patients.
6. **Phenotypes of enriched TCRs** — for the top clonotypes per group: cell-state
   composition, dominant state, signature scores vs background (log2FC +
   Mann-Whitney), tissue distribution.
7. **LN → tumour trafficking** — per patient: shared / LN-only / tumour-only
   clonotypes, Jaccard, Morisita-Horn, trafficking index (fraction of the
   tumour repertoire seen in the LN), group comparison (Kruskal-Wallis);
   per shared clonotype: frequency in each tissue, expansion in tumour,
   dominant state in each tissue, signature shifts LN→tumour (paired
   Wilcoxon per treatment), LN→tumour state-transition matrix.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Input data

```
data/
├── sample_metadata.csv     # columns: sample_id, patient, treatment, tissue
├── scrna/<sample_id>/      # 10x mtx dir (matrix.mtx.gz, barcodes, features) or .h5
└── tcr/<sample_id>.tsv     # AIRR rearrangement table (or CellRanger VDJ csv)
```

- `tissue` must contain the labels configured under
  `trafficking.ln_label` / `trafficking.tumor_label` (default `LN`, `tumor`).
- `treatment` is the grouping variable for enrichment (e.g. `control`,
  `antiPD1`, `combo`).
- Clonotypes are defined **globally** across samples, so the same TCR in a
  patient's lymph node and tumour gets the same `clone_id` — this is what
  enables the trafficking analysis.
- Cell barcodes are matched between GEX and TCR files per sample
  (`<sample_id>_<barcode>`); long enzyme/UMI suffixes are stripped.
- AIRR tables need at minimum: `cell_id`, `locus`, `junction_aa`, `junction`,
  `v_call`, `j_call`, `productive`, `consensus_count`, `duplicate_count`.
  Cellranger `filtered_contig_annotations.csv` works via `tcr_format: tenx`.

## Run

```bash
python run_pipeline.py --config config.yaml --stage all
# or stage by stage:
python run_pipeline.py --config config.yaml --stage qc integrate annotate tcr enrichment phenotype trafficking
```

All parameters (filtering thresholds, signature gene sets, clonotype metric,
group/tissue/patient column names, top-N, FDR alpha) live in `config.yaml`.

## Outputs (`results/`)

| File | Content |
|---|---|
| `tables/tcr_enrichment_<group>.csv` | clonotypes ranked by FDR / patient replication / log2FC for each treatment |
| `tables/enriched_tcr_phenotypes.csv` | state composition + signature scores of the top enriched TCRs per group |
| `tables/enriched_tcr_signature_tests.csv` | per-signature Mann-Whitney tests (clone vs background) with BH FDR |
| `tables/ln_tumor_overlap_per_patient.csv` | n shared clones, Jaccard, Morisita-Horn, trafficking index per patient |
| `tables/shared_clonotypes_ln_tumor.csv` | per shared clone: freq in LN/tumour, log2FC, expanded-in-tumour flag, dominant state in each tissue, signature shifts |
| `tables/overlap_group_comparison.csv` | overlap metrics aggregated by treatment with Kruskal-Wallis p |
| `tables/shared_clone_phenotype_shifts.csv` | paired LN→tumour signature shifts per treatment (Wilcoxon) |
| `tables/tcr_diversity.csv` | clonotype counts + Shannon diversity per patient/tissue |
| `processed/*.h5mu` | MuData snapshots after each stage (gex + airr modalities) |
| `figures/*.png` | QC violins, UMAPs (states/expansion), top-clone bars, diversity, enrichment volcanoes, phenotype heatmap/state bars, overlap boxes, shift heatmap, state transitions |

## Notes

- With few patients per group, Mann-Whitney/Kruskal tests are underpowered;
  the Fisher presence/absence test and effect sizes (log2FC, n_patients)
  complement them. Treat FDR as exploratory (`fdr_alpha: 0.25` default).
- For real data set `integration.method: scvi` (batch key = sample) and
  increase `scvi_max_epochs`.
- Clonotype definition uses exact CDR3 amino-acid identity by default;
  set `tcr.clonotype_metric` (e.g. `levenshtein`) for clustered clonotypes.
- A synthetic paired dataset generator (`scripts/make_synthetic_data.py`)
  reproduces the full directory layout for smoke tests.
