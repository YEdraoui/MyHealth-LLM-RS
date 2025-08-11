import argparse, torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.data_loader import MultimodalFundusDataset
from src.models import MultimodalClassifier
from src.utils import train_one_epoch, validate_one_epoch, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--train_csv', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='models/')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() or args.device=='mps' else 'cpu')

train_dataset = MultimodalFundusDataset(args.train_csv, img_dir='data/BRSET/fundus_photos', split='train')
val_dataset = MultimodalFundusDataset(args.train_csv, img_dir='data/BRSET/fundus_photos', split='val')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = MultimodalClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')
for epoch in range(1, args.epochs+1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, f"{args.output_dir}/phase2_best_model.pth")
        print(f"  ✅ Saved best model -> {args.output_dir}/phase2_best_model.pth")
