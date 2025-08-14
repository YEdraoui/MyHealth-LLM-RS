import os, csv, glob, random, json

root = "data/BRSET"
imgs_dir = os.path.join(root, "fundus_photos")
src_csv  = os.path.join(root, "labels_brset.csv")
out_csv  = os.path.join(root, "labels_normalized.csv")
splits_dir = "data/splits"
os.makedirs(splits_dir, exist_ok=True)

TARGET_LABELS = ["diabetic_retinopathy","macular_edema","amd","retinal_detachment","increased_cup_disc","other"]
META_KEEP = ["patient_id","patient_age","patient_sex","diabetes_time_y","insuline","exam_eye","diabetes","camera"]

# Index images
imgset = {os.path.basename(p) for p in glob.glob(os.path.join(imgs_dir, "*"))}
def to_jpg_name(x):
    x = (x or "").strip()
    if x.lower().endswith((".jpg",".jpeg",".png")):
        base = os.path.basename(x)
        return base if base.lower().endswith(".jpg") else os.path.splitext(base)[0] + ".jpg"
    return (x + ".jpg") if x else ""

# Load CSV
if not os.path.isfile(src_csv):
    raise SystemExit(f"CSV not found: {src_csv}")
with open(src_csv, newline='') as f:
    r = csv.reader(f)
    raw_header = next(r)
    header_map = {i: c.strip().lstrip("\ufeff").lower() for i, c in enumerate(raw_header)}
    rows = [{header_map[i]: row[i].strip() for i in range(len(row))} for row in r if row]

# Detect filename column
fname_key = None
for k in ("filename","image","img","image_id","file","path","image_path","img_name"):
    if k in rows[0]:
        fname_key = k; break
if fname_key is None:
    raise SystemExit("No filename column found in CSV.")

# Fill missing labels
existing_labels = [l for l in TARGET_LABELS if l in rows[0]]
missing_labels = [l for l in TARGET_LABELS if l not in existing_labels]

# Write normalized CSV
kept_meta = [m for m in META_KEEP if m in rows[0]]
out_header = ["filename"] + kept_meta + TARGET_LABELS
kept, missing_img = [], 0
for rec in rows:
    fid = to_jpg_name(rec.get(fname_key, ""))
    if fid not in imgset:
        missing_img += 1
        continue
    out = {"filename": fid}
    for m in kept_meta: out[m] = rec.get(m, "")
    for l in TARGET_LABELS:
        out[l] = "1" if rec.get(l, "") == "1" else "0"
    kept.append(out)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_header)
    w.writeheader()
    w.writerows(kept)

print(f"Normalized rows written: {len(kept)}")
print(f"Rows dropped (missing image): {missing_img}")
print("Kept metadata:", kept_meta)

# Patient-wise split if available
has_pid = all("patient_id" in r for r in kept) and any(r["patient_id"] for r in kept)
rng = random.Random(42)
if has_pid:
    pid_to_rows = {}
    for r in kept: pid_to_rows.setdefault(r["patient_id"], []).append(r)
    pids = list(pid_to_rows.keys()); rng.shuffle(pids)
    n = len(pids); n_train = int(0.8*n); n_val = int(0.1*n)
    train_p, val_p = set(pids[:n_train]), set(pids[n_train:n_train+n_val])
    split_rows = {"train":[], "val":[], "test":[]}
    for pid, rs in pid_to_rows.items():
        bucket = "train" if pid in train_p else "val" if pid in val_p else "test"
        split_rows[bucket].extend(rs)
else:
    rng.shuffle(kept)
    n = len(kept); n_train = int(0.8*n); n_val = int(0.1*n)
    split_rows = {
        "train": kept[:n_train],
        "val": kept[n_train:n_train+n_val],
        "test": kept[n_train+n_val:],
    }

# Save splits
for split, rows_ in split_rows.items():
    with open(os.path.join(splits_dir, f"{split}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader(); w.writerows(rows_)
    with open(os.path.join(splits_dir, f"{split}_files.txt"), "w") as f:
        for r in rows_: f.write(r["filename"] + "\n")

# Manifest
with open(os.path.join(splits_dir, "manifest.json"), "w") as f:
    json.dump({
        "root": os.path.abspath(root),
        "out_csv": os.path.abspath(out_csv),
        "counts": {k: len(v) for k,v in split_rows.items()},
        "meta_kept": kept_meta,
        "labels": TARGET_LABELS,
        "patient_wise": has_pid,
        "seed": 42
    }, f, indent=2)
print("Phase 0 complete.")

