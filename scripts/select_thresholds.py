import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve

ap=argparse.ArgumentParser()
ap.add_argument("--val-npz", default="reports/val_preds.npz")
ap.add_argument("--test-npz", default="reports/test_preds.npz")
ap.add_argument("--method", choices=["youden","maxf1"], default="youden")
ap.add_argument("--out-thresh", default="reports/thresholds.json")
ap.add_argument("--out-csv", default="reports/operating_point_metrics.csv")
args=ap.parse_args()

val=np.load(args.val_npz, allow_pickle=True); test=np.load(args.test_npz, allow_pickle=True)
labels = list(val["labels"])
y_true_v, y_prob_v = val["y_true"], val["y_prob"]
y_true_t, y_prob_t = test["y_true"], test["y_prob"]

def youden(yt, yp):
    fpr, tpr, thr = roc_curve(yt, yp)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k])

def maxf1(yt, yp):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.01,0.99,199):
        yb = (yp>=t).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(yt, yb, average="binary", zero_division=0)
        if f1>best_f1: best_f1, best_t = f1, t
    return float(best_t)

th = {}
for j,name in enumerate(labels):
    yt = y_true_v[:,j]; yp = y_prob_v[:,j]
    if len(set(yt))<2:
        th[name]=0.5
    else:
        th[name] = youden(yt, yp) if args.method=="youden" else maxf1(yt, yp)

# apply to test
rows=[]
for j,name in enumerate(labels):
    tt = th[name]
    yb = (y_prob_t[:,j] >= tt).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(y_true_t[:,j], yb, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_t[:,j], yb, labels=[0,1]).ravel()
    rows.append(dict(label=name, threshold=tt, precision=float(p), recall=float(r), f1=float(f1),
                     tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)))

with open(args.out_thresh, "w") as f: json.dump(th, f, indent=2)
pd.DataFrame(rows).to_csv(args.out_csv, index=False)
print("Saved", args.out_thresh, "and", args.out_csv)
