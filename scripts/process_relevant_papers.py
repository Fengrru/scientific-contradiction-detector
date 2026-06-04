"""
Download PDFs and extract text for truly relevant CoT papers.
Then extract claims and detect contradictions.
"""
import sys
sys.path.insert(0, "../src")

import pandas as pd
import os
import time
import requests
from tqdm import tqdm

from llm_client import LLMClient
from claim_extractor import ClaimExtractor
from data_cleaner import DataCleaner
from contradiction_detector import ContradictionDetector, manual_verification_sample
from rule_engine import RuleEngine, replace_llm_with_rules
from ontology import ONTOLOGY

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OUTPUT_DIR = "data"
PDF_DIR = "papers/pdfs"

os.makedirs(PDF_DIR, exist_ok=True)

print("=" * 60)
print("PROCESSING RELEVANT CoT PAPERS")
print("=" * 60)

# Load relevant papers
papers_df = pd.read_csv(os.path.join(OUTPUT_DIR, "relevant_core_papers.csv"))
print(f"Loaded {len(papers_df)} relevant papers")

# ============================================================
# STEP 1: Download missing PDFs
# ============================================================
print("\n--- STEP 1: Downloading PDFs ---")
pdfs_downloaded = 0
for idx, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="PDFs"):
    arxiv_id = row["arxiv_id"]
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    
    if os.path.exists(pdf_path):
        papers_df.at[idx, "pdf_exists"] = True
        continue
    
    # Try multiple PDF URL formats
    urls = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{arxiv_id}",
    ]
    downloaded = False
    for url in urls:
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200 and len(resp.content) > 10000:
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                downloaded = True
                break
        except:
            continue
        time.sleep(1)
    
    papers_df.at[idx, "pdf_exists"] = downloaded
    if downloaded:
        pdfs_downloaded += 1
    time.sleep(2)  # Rate limit

print(f"Downloaded {pdfs_downloaded} new PDFs")
papers_df = papers_df[papers_df.get("pdf_exists", False) == True]
print(f"Papers with PDFs: {len(papers_df)}")

# ============================================================
# STEP 2: Extract text from PDFs
# ============================================================
print("\n--- STEP 2: Extracting text ---")
import fitz

texts = []
for idx, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Extracting"):
    arxiv_id = row["arxiv_id"]
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # Find core sections
        full_text = full_text.replace("\n\n", "\n").replace("\t", " ").strip()
        core_sections = ["results", "discussion", "conclusion", "experiment", "evaluation", "finding"]
        start_idx = 0
        for section in core_sections:
            idx_pos = full_text.lower().find(section)
            if idx_pos != -1:
                start_idx = idx_pos
                break
        
        core_text = full_text[start_idx:] if start_idx > 0 else full_text
        texts.append(len(core_text) > 100)
        papers_df.at[idx, "full_text"] = core_text[:12000]
        papers_df.at[idx, "text_extracted"] = len(core_text) > 100
    except Exception as e:
        texts.append(False)
        papers_df.at[idx, "full_text"] = ""
        papers_df.at[idx, "text_extracted"] = False

print(f"Text extracted: {sum(texts)}/{len(papers_df)}")

# Save intermediate
papers_df.to_csv(os.path.join(OUTPUT_DIR, "relevant_papers_with_text.csv"), index=False)

# Filter to papers with text
valid_papers = papers_df[papers_df["text_extracted"] == True]
print(f"Valid papers for extraction: {len(valid_papers)}")

if len(valid_papers) == 0:
    print("ERROR: No papers with valid text!")
    sys.exit(1)

# ============================================================
# STEP 3: Extract claims via DeepSeek API
# ============================================================
print("\n--- STEP 3: Extracting claims ---")
client = LLMClient(api_key=API_KEY, model="deepseek-chat")
extractor = ClaimExtractor(api_client=client)
cleaner = DataCleaner(ontology=ONTOLOGY)

all_claims = []
t0 = time.time()

for idx, (_, row) in enumerate(valid_papers.iterrows()):
    arxiv_id = row["arxiv_id"]
    text = str(row.get("full_text", ""))[:12000]
    
    print(f"[{idx+1:2d}/{len(valid_papers)}] {arxiv_id} ", end="", flush=True)
    
    try:
        claims = extractor.extract_from_text(text, ONTOLOGY.concepts, ONTOLOGY.relations)
        for c in claims:
            c["arxiv_id"] = arxiv_id
            c["title"] = row.get("title", "")
            c["citations"] = row.get("citations", 0)
        all_claims.extend(claims)
        print(f"-> {len(claims)} claims")
    except Exception as e:
        print(f"ERROR: {e}")
    
    if (idx + 1) % 10 == 0:
        pd.DataFrame(all_claims).to_csv(os.path.join(OUTPUT_DIR, "all_claims_raw.csv"), index=False)

elapsed = time.time() - t0
raw_df = pd.DataFrame(all_claims)
raw_df.to_csv(os.path.join(OUTPUT_DIR, "all_claims_raw.csv"), index=False)

# Clean
cleaned_df = cleaner.process(raw_df)
cleaned_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_claims.csv"), index=False)
report = cleaner.generate_cleaning_report(raw_df, cleaned_df)

print(f"\nExtraction: {len(raw_df)} raw -> {len(cleaned_df)} cleaned ({report['retention_rate']:.1%})")
print(f"Time: {elapsed:.0f}s")

# ============================================================
# STEP 4: Contradiction Detection
# ============================================================
print("\n--- STEP 4: Contradiction detection ---")
detector = ContradictionDetector(
    citation_weight=0.6, confidence_weight=0.4,
    mock_mode=False, api_client=client
)

candidates = detector.generate_candidate_pairs(cleaned_df)
print(f"Generated {len(candidates)} candidate pairs")

if len(candidates) == 0:
    print("No candidates! Checking groups...")
    groups = cleaned_df.groupby(["subject","relation","object"]).size().sort_values(ascending=False)
    for (s,r,o), c in groups.head(15).items():
        print(f"  {c}x: {s} | {r} | {o}")
else:
    print(f"Classifying {len(candidates)} pairs...")
    classified = detector.classify_all_pairs(candidates)
    scored = detector.calculate_significance(classified)
    contradictions = detector.filter_contradictions(scored, min_significance=1.0)
    contradictions.to_csv(os.path.join(OUTPUT_DIR, "final_contradictions.csv"), index=False)
    
    print(f"\nContradictions found: {len(contradictions)}")
    if len(contradictions) > 0:
        print("\nTop contradictions:")
        for _, row in contradictions.head(10).iterrows():
            print(f"  [{row['contradiction_type']}] sig={row['significance_score']:.1f}")
            print(f"    {row['claim1_text'][:70]}")
            print(f"    {row['claim2_text'][:70]}")
        
        manual_verification_sample(contradictions, n_samples=min(100, len(contradictions)),
                                   output_path=os.path.join(OUTPUT_DIR, "manual_verification.csv"))

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"  Papers: {len(valid_papers)}")
print(f"  Claims: {len(cleaned_df)}")
print(f"  Candidates: {len(candidates)}")
print(f"  Contradictions: {len(contradictions) if len(candidates) > 0 else 0}")
print("=" * 60)
