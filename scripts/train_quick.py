import os, argparse, numpy as np, platform
import torch as T
from torch import nn, optim
from tqdm import tqdm
from src.utils import read_config, set_seed, get_device
from src.dataset_brset import make_loaders
from src.vision_backbone import build_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--splits-dir", default="data/splits_small")
    ap.add_argument("--backbone", default="resnet18")
    args=ap.parse_args()

    conf=read_config(); labels=conf["labels"]; ncls=len(labels)
    set_seed(42); dev=get_device()

    # macOS/MPS: avoid multiprocessing workers
    use_workers = 0 if (dev.type in ("mps","cpu") or platform.system()=="Darwin") else 2

    loaders=make_loaders(conf["data_root"], args.splits_dir, labels, args.img_size,
                         batch_size=args.batch_size, num_workers=use_workers)
    model=build_model(ncls, backbone=args.backbone, pretrained=True).to(dev)

    crit=nn.BCEWithLogitsLoss()
    opt=optim.AdamW(model.parameters(), lr=1e-4)
    scaler = T.amp.GradScaler(device="cuda" if dev.type=="cuda" else "cpu")

    model.train()
    for ep in range(args.epochs):
        pbar=tqdm(loaders["train"], desc=f"Quick Epoch {ep+1}/{args.epochs}")
        for xb,yb in pbar:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with T.amp.autocast(device_type=("cuda" if dev.type=="cuda" else "cpu"), enabled=(dev.type=="cuda")):
                logits = model(xb); loss = crit(logits, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))
    os.makedirs("models", exist_ok=True)
    T.save(model.state_dict(), "models/phase1_subset.pt")
    print("Saved models/phase1_subset.pt")

if __name__=="__main__":
    main()
