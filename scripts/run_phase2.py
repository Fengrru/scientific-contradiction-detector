"""Run Phase 2: Claim extraction with DeepSeek API on specified number of papers."""
import sys
sys.path.insert(0, "../src")

import pandas as pd
import time
from llm_client import LLMClient
from claim_extractor import ClaimExtractor
from data_cleaner import DataCleaner
from ontology import ONTOLOGY
import os

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MAX_PAPERS = 20  # Start with 20 papers
OUTPUT_DIR = "data"
BATCH_SAVE = 5

print("=" * 60)
print(f"PHASE 2: Claim Extraction (DeepSeek API, {MAX_PAPERS} papers)")
print("=" * 60)

client = LLMClient(api_key=API_KEY, model="deepseek-chat")
extractor = ClaimExtractor(api_client=client)
cleaner = DataCleaner(ontology=ONTOLOGY)

# Load papers with text
papers_df = pd.read_csv(os.path.join(OUTPUT_DIR, "papers_with_text.csv"))
papers_df = papers_df[papers_df["text_extracted"] == True].head(MAX_PAPERS)
print(f"Loaded {len(papers_df)} papers")

all_claims = []
start_time = time.time()

for idx, row in papers_df.iterrows():
    arxiv_id = row["arxiv_id"]
    paper_text = row.get("full_text", "")
    title = row.get("title", "")

    print(f"\n[{idx+1}/{len(papers_df)}] {arxiv_id} - {title[:70]}...")

    try:
        claims = extractor.extract_from_text(
            paper_text[:12000],
            ONTOLOGY.concepts,
            ONTOLOGY.relations
        )

        for claim in claims:
            claim["arxiv_id"] = arxiv_id
            claim["title"] = title
            claim["citations"] = row.get("citations", 0)

        all_claims.extend(claims)
        print(f"  -> {len(claims)} claims extracted")

        # Save intermediate results
        if (len(all_claims) > 0) and ((idx + 1) % BATCH_SAVE == 0 or idx == len(papers_df) - 1):
            temp_df = pd.DataFrame(all_claims)
            temp_path = os.path.join(OUTPUT_DIR, f"all_claims_raw.csv")
            temp_df.to_csv(temp_path, index=False)
            print(f"  [Saved {len(temp_df)} total claims to {temp_path}]")

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    time.sleep(0.5)  # Small delay between API calls

elapsed = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"Extraction complete: {len(all_claims)} claims from {MAX_PAPERS} papers")
print(f"Time: {elapsed:.0f}s ({elapsed/MAX_PAPERS:.1f}s/paper)")
print(f"{'=' * 60}")

# Save raw claims
raw_df = pd.DataFrame(all_claims)
raw_path = os.path.join(OUTPUT_DIR, "all_claims_raw.csv")
raw_df.to_csv(raw_path, index=False)
print(f"Raw claims saved to {raw_path}")

# Clean claims
if len(raw_df) > 0:
    cleaned_df = cleaner.process(raw_df)
    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_claims.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned claims: {len(cleaned_df)} (from {len(raw_df)} raw)")
    print(f"Saved to {cleaned_path}")
    
    report = cleaner.generate_cleaning_report(raw_df, cleaned_df)
    print(f"Retention rate: {report['retention_rate']:.1%}")
else:
    print("No claims extracted!")
