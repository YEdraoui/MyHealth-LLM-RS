import csv, random, os, sys
random.seed(42)
labels_csv, splits_dir = sys.argv[1], sys.argv[2]
with open(labels_csv, newline="") as f: rows = list(csv.DictReader(f))
from collections import defaultdict
by_pid = defaultdict(list)
for r in rows:
    pid = r.get("patient_id") or r["filename"].split("_")[0]
    by_pid[pid].append(r)
pids = list(by_pid.keys()); random.shuffle(pids)
n=len(pids); n_train=int(0.70*n); n_val=int(0.15*n)
train=set(pids[:n_train]); val=set(pids[n_train:n_train+n_val]); test=set(pids[n_train+n_val:])
splits={"train":[], "val":[], "test":[]}
for pid in train: splits["train"].extend(by_pid[pid])
for pid in val:   splits["val"].extend(by_pid[pid])
for pid in test:  splits["test"].extend(by_pid[pid])
os.makedirs(splits_dir, exist_ok=True)
for name,rows_ in splits.items():
    if not rows_: continue
    with open(os.path.join(splits_dir, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_[0].keys()); w.writeheader(); w.writerows(rows_)
print({k: len(v) for k,v in splits.items()})
