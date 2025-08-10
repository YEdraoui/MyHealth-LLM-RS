# === PHASE 0: Clean setup + move BRSET into the current repo ===
set -euo pipefail

# 0) Vars
REPO="$PWD"
DATA_DIR="$REPO/data"
BRSET_SRC="$REPO/BRSET"
BRSET_DST="$DATA_DIR/BRSET"
VENV_DIR="$REPO/venv"

# 1) Create data dir and MOVE BRSET inside it
mkdir -p "$DATA_DIR"
if [ -d "$BRSET_SRC" ] && [ ! -e "$BRSET_DST" ]; then
  echo "📦 Moving BRSET into $BRSET_DST ..."
  mv "$BRSET_SRC" "$BRSET_DST"
else
  echo "ℹ️ BRSET already moved or missing. (src: '$BRSET_SRC', dst: '$BRSET_DST')"
fi

# 2) Create expected links used by code
cd "$REPO"
rm -f data/brset data/brset_real/images || true
mkdir -p data/brset_real
ln -s "BRSET" "data/brset"
ln -s "BRSET/fundus_photos" "data/brset_real/images"

# 3) Fresh virtual env
echo "🐍 Creating fresh venv ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# 4) Install deps
if [ -f requirements.txt ]; then
  echo "📥 Installing requirements.txt ..."
  pip install -r requirements.txt
else
  echo "⚠️ requirements.txt not found. Installing minimal set ..."
  pip install torch torchvision pillow tqdm gdown beautifulsoup4
fi

# 5) Quick dataset check
echo "🖼  Sample images:"
ls -la "data/brset/fundus_photos" | head -n 10

# 6) Quick model smoke test
IMG="data/brset/fundus_photos/$(ls data/brset/fundus_photos | head -n 1)"
if [ -f "test_phase1_clean.py" ] && [ -f "$IMG" ]; then
  echo "🚀 Running demo on: $IMG"
  python test_phase1_clean.py --image "$IMG" || true
else
  echo "ℹ️ Skipping demo run (test_phase1_clean.py or image not found)."
fi

echo "✅ Phase 0 complete. You are ready to go!"
