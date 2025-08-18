import csv, os, sys, re

raw_csv = sys.argv[1]
out_csv = sys.argv[2]

# Map common header variants to our canonical names
CANON = {
    "filename": "filename",
    "image": "filename",
    "img": "filename",
    "patient_id": "patient_id",
    "patient": "patient_id",
    # disease labels (common aliases)
    "diabetic_retinopathy": "diabetic_retinopathy",
    "dr": "diabetic_retinopathy",
    "macular_edema": "macular_edema",
    "me": "macular_edema",
    "amd": "amd",
    "age_related_macular_degeneration": "amd",
    "retinal_detachment": "retinal_detachment",
    "rd": "retinal_detachment",
    "increased_cup_disc": "increased_cup_disc",
    "cup_disc": "increased_cup_disc",
    "cup_disc_increase": "increased_cup_disc",
    "other": "other",
    # optional demographics
    "age": "age",
    "gender": "gender",
    "sex": "gender",
}

LABELS = ["diabetic_retinopathy","macular_edema","amd","retinal_detachment","increased_cup_disc","other"]

def norm01(v):
    s = str(v).strip().lower()
    if s in ("1","true","yes","y"): return "1"
    if s in ("0","false","no","n",""): return "0"
    # numeric?
    try:
        return "1" if float(s)>0 else "0"
    except:
        return "0"

with open(raw_csv, newline="") as f:
    r = csv.DictReader(f)
    # build header map
    hmap = {}
    for h in r.fieldnames:
        key = re.sub(r"[^a-z0-9_]+","_", h.strip().lower())
        hmap[h] = CANON.get(key, key)

    rows = []
    for row in r:
        out = { "filename": row.get(next((k for k,v in hmap.items() if v=="filename"), ""), "").strip(),
                "patient_id": row.get(next((k for k,v in hmap.items() if v=="patient_id"), ""), "").strip() }
        out["age"] = row.get(next((k for k,v in hmap.items() if v=="age"), ""), "").strip()
        out["gender"] = row.get(next((k for k,v in hmap.items() if v=="gender"), ""), "").strip()
        for lab in LABELS:
            src = next((k for k,v in hmap.items() if v==lab), None)
            out[lab] = norm01(row.get(src,"0"))
        # basic sanity
        if not out["filename"]:
            continue
        rows.append(out)

# write normalized CSV
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename","patient_id","age","gender"]+LABELS)
    w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} rows → {out_csv}")
