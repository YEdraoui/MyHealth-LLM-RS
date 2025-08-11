#!/bin/bash
set -euo pipefail
source venv/bin/activate

python - <<'PY'
import pandas as pd
from pathlib import Path

csv_path = Path("data/BRSET/labels_brset.csv")
img_dir  = Path("data/BRSET/fundus_photos")

df = pd.read_csv(csv_path)
assert "filename" in df.columns, "CSV must contain a 'filename' column."

# Normalize filename -> ensure it has an extension
def with_ext(name: str) -> str:
    n = str(name)
    if n.lower().endswith((".jpg",".jpeg",".png")):
        return n
    return n + ".jpg"

df["filename"] = df["filename"].astype(str).map(with_ext)

# Keep only rows where the image actually exists
df["__exists"] = df["filename"].map(lambda x: (img_dir / x).is_file())
before = len(df)
df = df[df["__exists"]].drop(columns="__exists")
after = len(df)
missing = before - after

# Write back
df.to_csv(csv_path, index=False)

print("✅ BRSET validated & patched.")
print(f"   Total rows before: {before}")
print(f"   Rows dropped (missing images): {missing}")
print(f"   Total rows after:  {after}")
print("\n📄 Sample rows:")
print(df.head())
PY
