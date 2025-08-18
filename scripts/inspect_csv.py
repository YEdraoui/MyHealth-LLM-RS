import pandas as pd, sys
p=sys.argv[1]
df=pd.read_csv(p, nrows=5)
print("\n=== CSV HEADERS ===")
print(list(df.columns))
print("\n=== FIRST 5 ROWS (truncated) ===")
print(df.head(5).to_string(max_cols=40, max_colwidth=24))
