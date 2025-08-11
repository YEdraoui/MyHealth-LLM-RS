#!/bin/bash
set -e
EPOCHS=15
DEVICE=mps

echo "🚀 Training Phase 2 — Multimodal Fundus + Metadata"
python ophthalmology_llm/train_phase2.py \
    --epochs $EPOCHS \
    --device $DEVICE \
    --train_csv data/BRSET/labels_brset.csv \
    --output_dir models/
