"""
Scaled-up Phase 2: Extract claims from 50 domain-relevant papers using DeepSeek API.
"""
import sys
sys.path.insert(0, "../src")

import pandas as pd
import os
import time
from llm_client import LLMClient
from claim_extractor import ClaimExtractor
from data_cleaner import DataCleaner
from ontology import ONTOLOGY

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OUTPUT_DIR = "data"
MAX_PAPERS = 50

print("=" * 60)
print(f"PHASE 2: SCALED CLAIM EXTRACTION ({MAX_PAPERS} papers)")
print("=" * 60)

client = LLMClient(api_key=API_KEY, model="deepseek-chat")
extractor = ClaimExtractor(api_client=client)
cleaner = DataCleaner(ontology=ONTOLOGY)

# Load and filter papers
papers_df = pd.read_csv(os.path.join(OUTPUT_DIR, "core_papers.csv"))
text_df = pd.read_csv(os.path.join(OUTPUT_DIR, "papers_with_text.csv"))
df = papers_df.merge(text_df[["arxiv_id", "full_text", "text_extracted"]], on="arxiv_id", how="inner")
df = df[df["text_extracted"] == True]

# Filter by domain keywords
keywords = ["chain.of.thought", "math", "reasoning", "arithmetic", "gsm8k", "llm", "large language model", "problem.solv", "logic", "prompt"]
def is_relevant(abstract, title):
    text = (str(abstract) + " " + str(title)).lower()
    return any(k in text for k in keywords)

df["relevant"] = df.apply(lambda r: is_relevant(r["abstract"], r["title"]), axis=1)
relevant = df[df["relevant"]].head(MAX_PAPERS)
print(f"Selected {len(relevant)} relevant papers (from {len(df)} total with text)")

all_claims = []
t0 = time.time()

for idx, (_, row) in enumerate(relevant.iterrows()):
    arxiv_id = row["arxiv_id"]
    text = str(row.get("full_text", ""))[:12000]
    title = str(row.get("title", ""))[:80]
    
    print(f"[{idx+1:2d}/{len(relevant)}] {arxiv_id}: ...", end=" ", flush=True)
    
    try:
        claims = extractor.extract_from_text(text, ONTOLOGY.concepts, ONTOLOGY.relations)
        for c in claims:
            c["arxiv_id"] = arxiv_id
            c["title"] = title
            c["citations"] = row.get("citations", 0)
        all_claims.extend(claims)
        print(f"-> {len(claims)} claims")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Save every 10 papers
    if (idx + 1) % 10 == 0:
        pd.DataFrame(all_claims).to_csv(os.path.join(OUTPUT_DIR, "all_claims_raw.csv"), index=False)
        print(f"  [saved {len(all_claims)} total claims]")

elapsed = time.time() - t0

# Final save
raw_df = pd.DataFrame(all_claims)
raw_path = os.path.join(OUTPUT_DIR, "all_claims_raw.csv")
raw_df.to_csv(raw_path, index=False)

# Clean
cleaned_df = cleaner.process(raw_df)
cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_claims.csv")
cleaned_df.to_csv(cleaned_path, index=False)
report = cleaner.generate_cleaning_report(raw_df, cleaned_df)

print(f"\n{'=' * 60}")
print(f"PHASE 2 COMPLETE")
print(f"  Papers: {len(relevant)}")
print(f"  Raw claims: {len(raw_df)}")
print(f"  Cleaned claims: {len(cleaned_df)} (retention: {report['retention_rate']:.1%})")
print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(relevant),1):.1f}s/paper)")
print(f"  Saved to: {raw_path}, {cleaned_path}")
print(f"{'=' * 60}")
