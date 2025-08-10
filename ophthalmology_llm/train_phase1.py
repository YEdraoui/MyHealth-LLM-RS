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
