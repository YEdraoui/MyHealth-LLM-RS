import os, argparse, numpy as np
import torch as T
from src.utils import read_config, get_device
from src.dataset_brset import make_loaders
from src.vision_backbone import build_model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--ckpt", default="models/phase2_best.pt")
    ap.add_argument("--out-val", default="reports/val_preds.npz")
    ap.add_argument("--out-test", default="reports/test_preds.npz")
    args=ap.parse_args()

    conf=read_config(); labels=conf["labels"]; ncls=len(labels)
    dev=get_device()
    loaders=make_loaders(conf["data_root"], args.splits_dir, labels, args.img_size,
                         batch_size=args.batch_size, num_workers=0)

    model=build_model(ncls, pretrained=False).to(dev)
    sd=T.load(args.ckpt, map_location="cpu"); model.load_state_dict(sd, strict=False)
    model.eval()

    def dump(loader):
        ys=[]; ps=[]
        with T.no_grad():
            for xb,yb in loader:
                xb=xb.to(dev)
                probs=T.sigmoid(model(xb)).cpu().numpy()
                ys.append(yb.numpy()); ps.append(probs)
        if not ys: return None,None
        return np.concatenate(ys,0), np.concatenate(ps,0)

    os.makedirs("reports", exist_ok=True)
    if "val" in loaders:
        yt, yp = dump(loaders["val"])
        np.savez_compressed(args.out_val, y_true=yt, y_prob=yp, labels=np.array(labels))
        print("Saved", args.out_val)
    if "test" in loaders:
        yt, yp = dump(loaders["test"])
        np.savez_compressed(args.out_test, y_true=yt, y_prob=yp, labels=np.array(labels))
        print("Saved", args.out_test)

if __name__=="__main__":
    main()
