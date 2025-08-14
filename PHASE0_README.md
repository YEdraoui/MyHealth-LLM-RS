# Phase 0 — Dataset Preparation (Local Only)

This repository intentionally **excludes** dataset files. Phase 0 sets up the BRSET dataset locally and produces normalized labels + splits **without** committing any proprietary data.

## Steps
1) Place the **unzipped** dataset under the project root:
   - a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/
     - fundus_photos/
     - labels_brset.csv

2) Copy into canonical layout:
   - data/BRSET/fundus_photos/
   - data/BRSET/labels_brset.csv

3) Normalize labels + create splits:
   ```bash
   python phase0_prepare.py
This creates (locally, not versioned):

data/BRSET/labels_normalized.csv

data/splits/{train,val,test}.csv

data/splits/manifest.json

Notes

No images or CSVs are tracked in git.

All Phase 0 artifacts remain local by design.
