import os, json, argparse, numpy as np, platform
import torch as T
from torch import nn, optim
from tqdm import tqdm
from src.utils import read_config, set_seed, get_device, save_json
from src.dataset_brset import make_loaders
from src.vision_backbone import build_model
from src.metrics import summarize_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--head-hidden", type=int, default=0)
    args=ap.parse_args()

    conf=read_config(); labels=conf["labels"]; ncls=len(labels)
    set_seed(42); dev=get_device()

    # macOS/MPS: no multiprocessing DataLoader
    if dev.type in ("mps","cpu") or platform.system()=="Darwin":
        num_workers = 0
    else:
        num_workers = min(8, max(2, (os.cpu_count() or 8)//2))

    loaders=make_loaders(conf["data_root"], args.splits_dir, labels, args.img_size,
                         batch_size=args.batch_size, num_workers=num_workers)

    model=build_model(ncls, backbone=args.backbone, pretrained=True,
                      head_hidden=(args.head_hidden or None)).to(dev)
    crit=nn.BCEWithLogitsLoss()
    opt=optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.wd)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,args.epochs))
    scaler=T.amp.GradScaler(device="cuda" if dev.type=="cuda" else "cpu")

    best=-1.0; best_path="models/phase2_best.pt"; hist=[]
    os.makedirs("models", exist_ok=True)

    def eval_loader(loader):
        if loader is None: return None
        model.eval(); ys=[]; ps=[]; losses=[]
        with T.no_grad():
            for xb,yb in loader:
                xb,yb=xb.to(dev), yb.to(dev)
                logits=model(xb); loss=crit(logits,yb); losses.append(float(loss))
                ps.append(T.sigmoid(logits).cpu().numpy()); ys.append(yb.cpu().numpy())
        if not ys: return None
        import numpy as np
        y_true=np.concatenate(ys,0); y_prob=np.concatenate(ps,0)
        return float(np.mean(losses)), y_true, y_prob

    for ep in range(args.epochs):
        model.train()
        pbar=tqdm(loaders["train"], desc=f"Epoch {ep+1}/{args.epochs}")
        for xb,yb in pbar:
            xb,yb=xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with T.amp.autocast(device_type=("cuda" if dev.type=="cuda" else "cpu"), enabled=(dev.type=="cuda")):
                logits=model(xb); loss=crit(logits,yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))
        sch.step()

        v = eval_loader(loaders.get("val"))
        if v:
            vloss, y_true, y_prob = v
            mb = summarize_metrics(y_true, y_prob, labels)
            hist.append(dict(epoch=ep+1, val_loss=vloss, **mb))
            score = mb["macro_auROC"] if mb["macro_auROC"] is not None else -1.0
            if score>best:
                best=score; T.save(model.state_dict(), best_path)
            T.save(model.state_dict(), "models/phase2_last.pt")
            print(f"Val | loss={vloss:.4f} macroAUROC={mb[macro_auROC]} macroAUPR={mb[macro_AUPR]} pwAUROC={mb[prev_weighted_AUROC]}")

    save_json(dict(history=hist, best_macro_auroc=best, best_ckpt=best_path), "reports/phase2_train_history.json")
    print(f"Saved best to {best_path}")

if __name__=="__main__":
    main()
