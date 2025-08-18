import os, csv, random, argparse
random.seed(42)

ap=argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=2000)
ap.add_argument("--val",   type=int, default=400)
ap.add_argument("--test",  type=int, default=400)
ap.add_argument("--in-splits", default="data/splits")
ap.add_argument("--out", default="data/splits_small")
args=ap.parse_args()

def read_csv(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
def write_csv(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

splits={}
for s in ("train","val","test"):
    rows=read_csv(os.path.join(args.in_splits, f"{s}.csv"))
    random.shuffle(rows)
    k=getattr(args, s)
    splits[s]=rows[:min(k, len(rows))]

for s,rows in splits.items():
    if rows: write_csv(os.path.join(args.out,f"{s}.csv"), rows)
print({k: len(v) for k,v in splits.items()}, "->", args.out)
