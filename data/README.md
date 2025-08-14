# Data folder (not versioned)

The BRSET dataset (images + CSVs) is **not** committed to this repo.
Place your data at:

- data/BRSET/fundus_photos/
- data/BRSET/labels_brset.csv  (original vendor CSV, not tracked)
- data/BRSET/labels_normalized.csv (derived locally by phase0_prepare.py, not tracked)

To (re)create normalized labels + splits:
    python phase0_prepare.py
