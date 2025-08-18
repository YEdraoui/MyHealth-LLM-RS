import os, csv, json, yaml
conf = yaml.safe_load(open("configs/config.yaml"))
def load(p):
    with open(p, newline="") as f: 
        return list(csv.DictReader(f))
def prevalence(rows, labels):
    n = max(1, len(rows)); out={}
    for l in labels:
        pos = sum(1 for r in rows if str(r.get(l,"0")).strip() in ("1","True","true"))
        out[l] = round(pos/n, 4)
    return out
img_dir = os.path.join(conf["data_root"], "fundus_photos")
labels = conf["labels"]; sp = conf["splits_dir"]
paths = {k: os.path.join(sp, f"{k}.csv") for k in ("train","val","test")}
rows = {k: load(v) for k,v in paths.items() if os.path.isfile(v)}
counts = {k: len(v) for k,v in rows.items()}
prev = {k: prevalence(v, labels) for k,v in rows.items()}
missing = {k: sum(0 if os.path.isfile(os.path.join(img_dir, r["filename"])) else 1 for r in v)
           for k,v in rows.items()}
def leaks(a,b):
    A=set(r.get("patient_id","") for r in a); B=set(r.get("patient_id","") for r in b)
    return len(A & B)
leak = {
  "pid_overlap_train_val": leaks(rows.get("train",[]), rows.get("val",[])),
  "pid_overlap_train_test": leaks(rows.get("train",[]), rows.get("test",[])),
  "pid_overlap_val_test": leaks(rows.get("val",[]), rows.get("test",[])),
}
out = dict(data_root=conf["data_root"], counts=counts, prevalence=prev,
           missing_files_in_splits=missing, leakage=leak, labels=labels)
os.makedirs("reports", exist_ok=True)
open("reports/phase1_eda_report.json","w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print("\nSaved → reports/phase1_eda_report.json")
