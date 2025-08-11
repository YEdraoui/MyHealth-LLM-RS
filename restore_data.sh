#!/bin/bash
set -e

echo "📂 Restoring dataset structure..."

SRC_DIR="data/BRSET/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/fundus_photos"
DEST_DIR="data/BRSET/fundus_photos"

# 1. Move extracted images in batches
if [ -d "$SRC_DIR" ]; then
    mkdir -p "$DEST_DIR"
    echo "🚚 Moving images in batches..."
    find "$SRC_DIR" -type f -name "*.jpg" -print0 | xargs -0 -I{} mv "{}" "$DEST_DIR"/
    echo "✅ Moved all fundus images into $DEST_DIR/"
else
    echo "❌ Fundus photos source folder not found — please unzip dataset again."
fi

# 2. Copy labels CSV if missing
if [ ! -f "data/BRSET/labels_brset.csv" ] && [ -f "data/BRSET/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/labels_brset.csv" ]; then
    cp data/BRSET/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/labels_brset.csv data/BRSET/
    echo "✅ Copied labels CSV."
fi

# 3. Restore models if backup exists
mkdir -p models
for model in phase1_best_model.pth phase2_best_model.pth; do
    if [ ! -f "models/$model" ]; then
        if [ -f "backups/$model" ]; then
            cp "backups/$model" models/
            echo "✅ Restored $model from backups/"
        else
            echo "⚠️ $model missing — needs retraining."
        fi
    else
        echo "✅ $model already exists."
    fi
done

echo "🎯 Done! Run your Phase 2 test again."

