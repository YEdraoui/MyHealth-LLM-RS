#!/usr/bin/env bash
# === Phase 1: Clean repo + train 6-label fundus classifier on BRSET ===
# Run this from: ~/Desktop/Internship/LLM\ Model/medication_recommendation_model
set -euo pipefail

# ---------- Paths ----------
ROOT="$PWD"
PROJ="ophthalmology_llm"
REPO="$ROOT/$PROJ"
BRSET_DIR="$ROOT/data/BRSET"
IMAGES_DIR="$BRSET_DIR/fundus_photos"
LABELS_CSV="$BRSET_DIR/labels_brset.csv"

# ---------- Checks ----------
if [ ! -d "$IMAGES_DIR" ] || [ ! -f "$LABELS_CSV" ]; then
  echo "❌ BRSET not found. Expecting:"
  echo "   - $IMAGES_DIR"
  echo "   - $LABELS_CSV"
  exit 1
fi

echo "✅ Found BRSET dataset."

# ---------- Repo skeleton ----------
echo "📁 Creating clean repo at: $REPO"
rm -rf "$REPO"
mkdir -p "$REPO"/{src,llm,models,notebooks}
mkdir -p "$REPO/data"
# Link BRSET inside the repo for convenience
ln -s "$BRSET_DIR" "$REPO/data/BRSET"

# ---------- Requirements ----------
cat > "$REPO/requirements.txt" <<'REQ'
torch
torchvision
pillow
tqdm
pandas
numpy
scikit-learn
matplotlib
REQ

# ---------- Model ----------
cat > "$REPO/src/model.py" <<'PY'
import torch
import torch.nn as nn
from torchvision import models

NUM_LABELS = 6

class FundusClassifier(nn.Module):
    def __init__(self, num_labels: int = NUM_LABELS):
        super().__init__()
        # ResNet50 backbone (ImageNet)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        # Replace head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_labels)
        )

    def forward(self, x):
        return self.backbone(x)

def create_model():
    return FundusClassifier(NUM_LABELS)
PY

# ---------- Data loader / splitting ----------
cat > "$REPO/src/data_loader.py" <<'PY'
import os
import pandas as pd
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

# Expected label columns in labels_brset.csv (0/1 per image)
LABEL_COLUMNS = [
    "diabetic_retinopathy",
    "macular_edema",
    "amd",
    "retinal_detachment",
    "increased_cup_disc",
    "other",
]

class BRSETDataset(Dataset):
    def __init__(self, images_dir: str, df: pd.DataFrame, transform=None):
        self.images_dir = images_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Validate columns
        missing = [c for c in LABEL_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns in CSV: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[LABEL_COLUMNS].values.astype("float32"))

        if self.transform:
            image = self.transform(image)

        return image, labels

def make_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

def ensure_splits(labels_csv: str, out_dir: str, seed: int = 42) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    test_csv  = os.path.join(out_dir, "test.csv")

    if all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        return train_csv, val_csv, test_csv

    df = pd.read_csv(labels_csv)
    if "filename" not in df.columns:
        # Try common alternatives
        alt = "image" if "image" in df.columns else None
        if alt is None:
            raise ValueError("CSV must have 'filename' column (or 'image').")
        df = df.rename(columns={alt: "filename"})

    # Basic random split (multi-label stratification is out-of-scope here)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, shuffle=True)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=seed, shuffle=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv

def make_loaders(images_dir: str, split_dir: str, batch_size: int = 32, num_workers: int = 0):
    import pandas as pd
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(split_dir, "val.csv"))

    train_ds = BRSETDataset(images_dir, train_df, transform=make_transforms(train=True))
    val_ds   = BRSETDataset(images_dir, val_df,   transform=make_transforms(train=False))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ld, val_ld
PY

# ---------- Training ----------
cat > "$REPO/train_phase1.py" <<'PY'
import os, argparse, time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.model import create_model
from src.data_loader import ensure_splits, make_loaders, LABEL_COLUMNS

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BRSET", type=str)
    ap.add_argument("--images", default="fundus_photos", type=str)
    ap.add_argument("--labels_csv", default="labels_brset.csv", type=str)
    ap.add_argument("--splits_dir", default="splits", type=str)
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument("--out", default="models/phase1_best_model.pth", type=str)
    args = ap.parse_args()

    data_root = args.data_root
    images_dir = os.path.join(data_root, args.images)
    labels_csv = os.path.join(data_root, args.labels_csv)
    splits_dir = os.path.join(data_root, args.splits_dir)

    # Ensure splits exist
    train_csv, val_csv, test_csv = ensure_splits(labels_csv, splits_dir)

    # Loaders
    train_ld, val_ld = make_loaders(images_dir, splits_dir, batch_size=args.batch_size)

    # Model
    device = torch.device(args.device)
    model = create_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_ld, device, criterion, optimizer)
        va_loss = eval_one_epoch(model, val_ld, device, criterion)
        scheduler.step()
        dt = time.time() - t0
        print(f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  ({dt:.1f}s)")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.out)
            print(f"  ✅ Saved best model -> {args.out}")

    print("Done.")
    print(f"Best val loss: {best_val:.4f}")
    print("NOTE: For thorough evaluation, write a separate test script using the saved state dict.")

if __name__ == "__main__":
    main()
PY

# ---------- Clinical LLM ----------
cat > "$REPO/llm/clinical_llm.py" <<'PY'
class ClinicalOphthalmologyLLM:
    def __init__(self):
        self.conditions = [
            'diabetic_retinopathy', 'macular_edema', 'amd',
            'retinal_detachment', 'increased_cup_disc', 'other'
        ]
        self.recommendations_db = {
            'diabetic_retinopathy': 'Optimize diabetes control and refer to ophthalmology.',
            'macular_edema': 'Consider OCT and anti-VEGF evaluation.',
            'amd': 'Monitor drusen, consider AREDS and retina clinic follow-up.',
            'retinal_detachment': 'Urgent retina consult recommended.',
            'increased_cup_disc': 'Assess IOP and glaucoma risk; schedule evaluation.'
        }

    def format_findings(self, probs, thr=0.5):
        F = []
        urgent = []
        for cond, p in zip(self.conditions, probs):
            if p >= thr:
                conf = "high" if p >= 0.8 else "moderate" if p >= 0.65 else "low"
                F.append(f"- {cond.replace('_',' ').title()}: {conf} ({p:.2f})")
                if cond in {"retinal_detachment"} and p >= 0.5:
                    urgent.append(cond)
        if not F:
            F = ["- No strong abnormality detected (all below threshold)"]
        return F, urgent

    def summarize(self, patient, probs):
        findings, urgent = self.format_findings(probs)
        recs = []

        if urgent:
            recs.append("🚨 URGENT: Findings may require immediate retina evaluation.")
            for u in urgent:
                if u in self.recommendations_db:
                    recs.append(f"• {self.recommendations_db[u]}")

        # Generic
        if not recs:
            recs.append("• Routine follow-up and risk-factor optimization as indicated.")

        age = patient.get("age", "unknown")
        if isinstance(age, (int, float)) and age >= 60:
            recs.append("• Age ≥60: consider regular retinal screening.")

        return (
            "📋 OPHTHALMOLOGY AI SUMMARY\n\n"
            f"Patient: age={patient.get('age','?')}, sex={patient.get('sex','?')}\n\n"
            "RETINAL FINDINGS:\n" + "\n".join(findings) + "\n\n" +
            "RECOMMENDATIONS:\n" + "\n".join(recs) + "\n\n" +
            "⚠️ This AI output requires clinical validation."
        )

def create_clinical_llm():
    return ClinicalOphthalmologyLLM()
PY

# ---------- Inference / demo ----------
cat > "$REPO/test_phase1_clean.py" <<'PY'
import argparse, os
import torch
from PIL import Image
import torchvision.transforms as T
from src.model import create_model
from llm.clinical_llm import create_clinical_llm

LABELS = [
    "diabetic_retinopathy",
    "macular_edema",
    "amd",
    "retinal_detachment",
    "increased_cup_disc",
    "other",
]

def load_image(path):
    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str, help="Path to a fundus image (.jpg)")
    ap.add_argument("--weights", default="models/phase1_best_model.pth", type=str)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = create_model()
    if os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print("✅ Loaded weights:", args.weights)
    else:
        print("⚠️ Weights not found, using randomly initialized model.")

    model.eval()
    x = load_image(args.image)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    print("\n📊 Probabilities:")
    for k, p in zip(LABELS, probs):
        print(f"  {k}: {p:.3f}")

    # simple patient
    llm = create_clinical_llm()
    patient = {"age": 67, "sex": "F"}
    print("\n" + llm.summarize(patient, probs))

if __name__ == "__main__":
    main()
PY

