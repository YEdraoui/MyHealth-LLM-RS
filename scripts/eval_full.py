import os, json, argparse, numpy as np
import torch as T
from src.utils import read_config, get_device, save_json
from src.dataset_brset import make_loaders
from src.vision_backbone import build_model
from src.metrics import summarize_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--ckpt", default="models/phase2_best.pt")
    args=ap.parse_args()

    conf=read_config(); labels=conf["labels"]; ncls=len(labels)
    dev=get_device(); tf_img=args.img_size
    loaders=make_loaders(conf["data_root"], args.splits_dir, labels, tf_img,
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

    out={"labels": labels, "checkpoint": args.ckpt, "img_size": tf_img, "batch_size": args.batch_size}
    if "val" in loaders:
        yt, yp = dump(loaders["val"])
        out["val"]=summarize_metrics(yt, yp, labels)
    if "test" in loaders:
        yt, yp = dump(loaders["test"])
        out["test"]=summarize_metrics(yt, yp, labels)

    save_json(out, "reports/phase2_eval_report.json")
    print(json.dumps(out, indent=2))
    print("Saved reports/phase2_eval_report.json")

if __name__=="__main__":
    main()
