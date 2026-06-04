"""
Full pipeline re-run with fixes.
Uses existing 42 relevant papers + merges with 200-paper text dataset.
"""
import sys, os, time
sys.path.insert(0, "src")
import pandas as pd
import requests
from tqdm import tqdm
import fitz
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
print("FULL PIPELINE RE-RUN (fixed prompts + cleaner)")
print("=" * 60)

# ============================================================
# STEP 1: Load and merge paper data
# ============================================================
print("\n--- STEP 1: Loading papers ---")
relevant = pd.read_csv(os.path.join(OUTPUT_DIR, "relevant_core_papers.csv"))
full_texts = pd.read_csv(os.path.join(OUTPUT_DIR, "papers_with_text.csv"))
# Ensure arxiv_id is string type for merge
relevant["arxiv_id"] = relevant["arxiv_id"].astype(str)
full_texts["arxiv_id"] = full_texts["arxiv_id"].astype(str)
print(f"Relevant papers: {len(relevant)}")
print(f"Full text dataset: {len(full_texts)} papers")

# Merge existing full_text where available
# Also try to match papers (some might have slightly different arXiv IDs)
merged = relevant.merge(full_texts[["arxiv_id", "full_text", "text_extracted"]], on="arxiv_id", how="left")

# For papers without existing text, check if their PDF exists already
merged["has_pdf"] = merged["arxiv_id"].apply(lambda x: os.path.exists(os.path.join(PDF_DIR, f"{x}.pdf")))
# text_extracted might be NaN from left merge, treat as False
merged["text_extracted"] = merged["text_extracted"].fillna(False).astype(bool)
has_text = merged["full_text"].notna() & (merged["full_text"].astype(str).str.len() > 200)
print(f"With existing text: {has_text.sum()}")
print(f"With existing PDF: {merged['has_pdf'].sum()}")

# ============================================================
# STEP 2: Download missing PDFs
# ============================================================
need_download = merged[~has_text & ~merged["has_pdf"]]
print(f"\n--- STEP 2: Download {len(need_download)} missing PDFs ---")
for idx, row in tqdm(need_download.iterrows(), total=len(need_download)):
    arxiv_id = row["arxiv_id"]
    for url in [f"https://arxiv.org/pdf/{arxiv_id}.pdf", f"https://arxiv.org/pdf/{arxiv_id}"]:
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200 and len(resp.content) > 10000:
                with open(os.path.join(PDF_DIR, f"{arxiv_id}.pdf"), "wb") as f:
                    f.write(resp.content)
                merged.at[idx, "has_pdf"] = True
                break
        except:
            continue
    time.sleep(1)

# ============================================================
# STEP 3: Extract text for papers missing it
# ============================================================
need_text = merged[~has_text & merged["has_pdf"]]
print(f"\n--- STEP 3: Extract text for {len(need_text)} papers ---")
for idx, row in tqdm(need_text.iterrows(), total=len(need_text)):
    arxiv_id = row["arxiv_id"]
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        text = text.replace("\n\n", "\n").replace("\t", " ").strip()
        for sec in ["results", "discussion", "conclusion", "experiment", "evaluation", "finding"]:
            pos = text.lower().find(sec)
            if pos != -1:
                text = text[pos:]
                break
        merged.at[idx, "full_text"] = text[:12000]
        merged.at[idx, "text_extracted"] = len(text) > 200
    except:
        merged.at[idx, "full_text"] = ""
        merged.at[idx, "text_extracted"] = False

# Final set: papers with text (either from existing text_df or newly extracted)
merged["full_text"] = merged["full_text"].fillna("").astype(str)
papers_df = merged[merged["text_extracted"] == True].copy()
print(f"\nPapers ready for extraction: {len(papers_df)}")
papers_df.to_csv(os.path.join(OUTPUT_DIR, "relevant_papers_with_text.csv"), index=False)

# ============================================================
# STEP 4: Claim extraction via DeepSeek API
# ============================================================
print("\n--- STEP 4: Claim extraction ---")
client = LLMClient(api_key=API_KEY, model="deepseek-chat")
extractor = ClaimExtractor(api_client=client)
cleaner = DataCleaner(ontology=ONTOLOGY)

all_claims = []
t0 = time.time()

for idx, (_, row) in enumerate(papers_df.iterrows()):
    arxiv_id = row["arxiv_id"]
    text = str(row.get("full_text", ""))[:12000]
    print(f"[{idx+1:3d}/{len(papers_df)}] {arxiv_id} ", end="", flush=True)
    try:
        claims = extractor.extract_from_text(text, ONTOLOGY.concepts, ONTOLOGY.relations)
        for c in claims:
            c["arxiv_id"] = arxiv_id
            c["title"] = str(row.get("title", ""))[:80]
            c["citations"] = row.get("citations", 0)
        all_claims.extend(claims)
        print(f"{len(claims)} claims")
    except Exception as e:
        print(f"ERROR: {e}")

elapsed = time.time() - t0
raw_df = pd.DataFrame(all_claims)
raw_df.to_csv(os.path.join(OUTPUT_DIR, "all_claims_raw.csv"), index=False)

cleaned_df = cleaner.process(raw_df)
cleaned_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_claims.csv"), index=False)
report = cleaner.generate_cleaning_report(raw_df, cleaned_df)
print(f"\nExtraction: {len(raw_df)} raw -> {len(cleaned_df)} cleaned ({report['retention_rate']:.1%}) in {elapsed:.0f}s")
print("Top subjects:", cleaned_df["subject"].value_counts().head(5).to_dict())
print("Top relations:", cleaned_df["relation"].value_counts().head(5).to_dict())

# ============================================================
# STEP 5: Contradiction detection (hybrid: rule engine + LLM)
# ============================================================
print("\n--- STEP 5: Contradiction detection ---")
detector = ContradictionDetector(
    citation_weight=0.6, confidence_weight=0.4,
    mock_mode=False, api_client=client
)

candidates = detector.generate_candidate_pairs(cleaned_df)
candidates.to_csv(os.path.join(OUTPUT_DIR, "candidate_contradictions.csv"), index=False)
print(f"Candidate pairs: {len(candidates)}")

if len(candidates) > 0:
    engine = RuleEngine()
    summary = engine.get_rule_summary()
    print(f"Rules loaded: {summary['total_rules']}")
    
    # LLM classification
    print("Classifying with DeepSeek API...")
    classified = detector.classify_all_pairs(candidates)
    
    # Rule engine hybrid
    print("Merging rule engine...")
    hybrid = replace_llm_with_rules(classified, engine)
    classified["contradiction_type"] = hybrid["final_type"]
    
    # Score and filter
    scored = detector.calculate_significance(classified)
    contradictions = detector.filter_contradictions(scored, min_significance=0.0)
    contradictions.to_csv(os.path.join(OUTPUT_DIR, "final_contradictions.csv"), index=False)
    
    print(f"\nContradictions: {len(contradictions)}")
    print("Type distribution:")
    print(contradictions["contradiction_type"].value_counts())
    
    print("\nTop contradictions:")
    for _, r in contradictions.head(10).iterrows():
        print(f"  [{r['contradiction_type'][:35]:35s}] sig={r['significance_score']:6.1f}")
        print(f"    {r['claim1_text'][:55]} ({r['claim1_arxiv']})")
        print(f"    {r['claim2_text'][:55]} ({r['claim2_arxiv']})")
    
    manual_verification_sample(contradictions, n_samples=min(100, len(contradictions)),
                               output_path=os.path.join(OUTPUT_DIR, "manual_verification.csv"))
else:
    print("No candidate pairs found.")
    groups = cleaned_df.groupby(["subject","relation","object"]).size().sort_values(ascending=False)
    print("Groups requiring 2+ for a pair:")
    multi = groups[groups >= 2]
    for (s, r, o), c in multi.head(20).items():
        print(f"  {c}x: {s} | {r} | {o}")

# ============================================================
# STEP 6: Generate paper
# ============================================================
print("\n--- STEP 6: Paper generation ---")
from paper_generator import PaperGenerator, generate_paper_summary
if os.path.exists(os.path.join(OUTPUT_DIR, "final_contradictions.csv")):
    cdf = pd.read_csv(os.path.join(OUTPUT_DIR, "final_contradictions.csv"))
    if len(cdf) > 0:
        gen = PaperGenerator(domain="Chain-of-Thought Mathematical Reasoning in LLMs", target_venue="arXiv")
        tex = gen.generate_full_paper(cdf, cleaned_df, papers_df)
        os.makedirs("papers", exist_ok=True)
        with open("papers/contradiction_paper.tex", "w", encoding="utf-8") as f:
            f.write(tex)
        summary = generate_paper_summary(cdf)
        with open("results/README_SUMMARY.md", "w", encoding="utf-8") as f:
            f.write(summary)
        print("Paper and summary saved.")

print("\n" + "=" * 60)
print("COMPLETE")
print(f"  Papers: {len(papers_df)}")
print(f"  Claims: {len(cleaned_df)}")
print(f"  Pairs: {len(candidates)}")
print(f"  Contradictions: {len(contradictions) if len(candidates)>0 else 0}")
print("=" * 60)
