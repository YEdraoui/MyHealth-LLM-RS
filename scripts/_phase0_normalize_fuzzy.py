import re, csv, sys

raw_csv, out_csv = sys.argv[1], sys.argv[2]

# Canonical label set
LABELS = ["diabetic_retinopathy","macular_edema","amd","retinal_detachment","increased_cup_disc","other"]

# Synonyms/fuzzy patterns per field
PAT = {
  "filename": [r"^file(name)?$", r"^image(_?name)?$", r"^img$", r"^img_?path$", r"^path$", r"^image_id$"],
  "patient_id": [r"^patient(_?id)?$", r"^subject(_?id)?$", r"^pid$", r"^id_patient$"],
  "age": [r"^age$"],
  "gender": [r"^gender$", r"^sex$"],

  "diabetic_retinopathy": [r"^diabetic[_ ]?retinopathy$", r"^dr$"],
  "macular_edema": [r"^macular[_ ]?edema$", r"^me$"],
  "amd": [r"^amd$", r"^age[_ ]?related[_ ]?macular[_ ]?degeneration$"],
  "retinal_detachment": [r"^retinal[_ ]?detachment$", r"^rd$"],
  "increased_cup_disc": [r"^increased[_ ]?cup[_ ]?disc$", r"^cup[_ ]?disc(_?increase)?$", r"^glaucoma$", r"^cdr$", r"^cup[_ ]?to[_ ]?disc(_?ratio)?$"],
  "other": [r"^other$", r"^others?$"]
}

def norm_key(s:str)->str:
    return re.sub(r"[^a-z0-9]+","_", s.strip().lower())

def match_col(cols, patterns):
    for c in cols:
        ck = norm_key(c)
        for p in patterns:
            if re.fullmatch(p, ck):
                return c
    return None

def to01(v):
    s = str(v).strip().lower()
    if s in ("1","true","yes","y","pos","positive"): return "1"
    if s in ("0","false","no","n","neg","negative",""): return "0"
    try:
        return "1" if float(s)>0 else "0"
    except:
        return "0"

with open(raw_csv, newline="") as f:
    r = csv.DictReader(f)
    cols = list(r.fieldnames)

    colmap = {}
    for field, pats in PAT.items():
        m = match_col(cols, pats)
        if m: colmap[field]=m

    # Require filename; if missing, try image_id or similar
    if "filename" not in colmap:
        # best-effort: look for any column that ends with .jpg-like content
        sample = next(iter(r), None)
        if sample:
            for c in cols:
                if str(sample[c]).lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp")):
                    colmap["filename"]=c
                    break
        f.seek(0); r = csv.DictReader(f)

    out_rows = []
    for row in r:
        fn = row.get(colmap.get("filename",""), "").strip()
        if not fn: 
            continue
        pid = row.get(colmap.get("patient_id",""), "").strip() or fn.split("_")[0]
        age = row.get(colmap.get("age",""), "").strip()
        gender = row.get(colmap.get("gender",""), "").strip()

        rec = {"filename": fn, "patient_id": pid, "age": age, "gender": gender}
        for lab in LABELS:
            src = colmap.get(lab, None)
            rec[lab] = to01(row.get(src, "0"))
        out_rows.append(rec)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename","patient_id","age","gender"]+LABELS)
    w.writeheader(); w.writerows(out_rows)

print(f"✅ Normalized {len(out_rows)} rows → {out_csv}")
