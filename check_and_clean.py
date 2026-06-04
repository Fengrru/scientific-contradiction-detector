import sys
sys.path.insert(0, "src")
import pandas as pd
from data_cleaner import DataCleaner
from ontology import ONTOLOGY

raw = pd.read_csv("data/all_claims_raw.csv")
print("Raw claims:", len(raw))

cleaner = DataCleaner(ontology=ONTOLOGY)
cleaned = cleaner.process(raw)
cleaned.to_csv("data/cleaned_claims.csv", index=False)

report = cleaner.generate_cleaning_report(raw, cleaned)
retention = report["retention_rate"]
print("Cleaned:", len(cleaned), f"(retention: {retention:.1%})")

groups = cleaned.groupby(["subject", "relation", "object"]).size().sort_values(ascending=False)
print("\nUnique groups:", len(groups))
print("Multi-claim groups:")
for (s, r, o), c in groups[groups >= 2].items():
    print(f"  {c}x: {s} | {r} | {o}")
