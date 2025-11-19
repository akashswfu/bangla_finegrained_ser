import csv, collections

path = "data/manifest_split.csv"   # <-- current file
cnt = collections.Counter()

with open(path, newline='', encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        cnt[row.get("split","").strip().lower()] += 1

print("Split counts in", path)
for k,v in cnt.items():
    print(f"  {repr(k)}: {v}")