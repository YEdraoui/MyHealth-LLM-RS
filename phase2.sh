#!/bin/bash
set -euo pipefail

# Activate venv
source venv/bin/activate

echo "=== Phase 2: Dataset merge + training ==="

# 1) Paths
BRSET_CSV="data/BRSET/labels_brset.csv"
BRSET_IMG="data/BRSET/fundus_photos"
OTHER_CSV="data/OTHER/labels_other.csv"          # change if dataset is different
OTHER_IMG="data/OTHER/fundus_photos"

MERGED_CSV="data/merged_labels.csv"
SPLITS_DIR="data/splits_phase2"

# 2) Merge datasets (if OTHER exists)
python - <<'PY'
import pandas as pd
from pathlib import Path

brset_csv = Path("data/BRSET/labels_brset.csv")
other_csv = Path("data/OTHER/labels_other.csv")

if not other_csv.exists():
    print("⚠️ No additional dataset found — using BRSET only.")
    df = pd.read_csv(brset_csv)
else:
    df1 = pd.read_csv(brset_csv)
    df2 = pd.read_csv(other_csv)
    # Ensure same columns
    missing_cols = [c for c in df1.columns if c not in df2.columns]
    for c in missing_cols:
        df2[c] = None
    df2 = df2[df1.columns]  # reorder
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"✅ Merged datasets: {len(df1)} + {len(df2)} rows")

df.to_csv("data/merged_labels.csv", index=False)
print(f"💾 Saved merged CSV -> data/merged_labels.csv  ({len(df)} rows)")
PY

# 3) Run training for Phase 2
python ophthalmology_llm/train_phase2.py \
    --labels $MERGED_CSV \
    --splits_dir $SPLITS_DIR \
    --epochs 10 \
    --device mps

echo "✅ Phase 2 complete."
