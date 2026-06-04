"""
Full pipeline runner: filter relevant papers, extract claims, detect contradictions.
All powered by DeepSeek API.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
import os
import time
from llm_client import LLMClient
from claim_extractor import ClaimExtractor
from data_cleaner import DataCleaner
from contradiction_detector import ContradictionDetector, manual_verification_sample
from rule_engine import RuleEngine, replace_llm_with_rules
from ontology import ONTOLOGY

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OUTPUT_DIR = "data"

print("=" * 60)
print("SCIENTIFIC CONTRADICTION DETECTION - FULL PIPELINE")
print("=" * 60)

client = LLMClient(api_key=API_KEY, model="deepseek-chat")

# ============================================================
# STEP 1: Filter relevant papers using DeepSeek
# ============================================================
print("\n--- STEP 1: Filtering papers for domain relevance ---")
papers_df = pd.read_csv(os.path.join(OUTPUT_DIR, "core_papers.csv"))

# Quick relevance check: filter by abstract keywords first
keywords = ["chain.of.thought", "math", "reasoning", "arithmetic", "gsm8k", "llm", "large language model"]
def is_relevant(abstract, title):
    text = (abstract + " " + title).lower()
    return any(k in text for k in keywords)

papers_df["keyword_match"] = papers_df.apply(
    lambda r: is_relevant(str(r["abstract"]), str(r["title"])), axis=1
)
relevant = papers_df[papers_df["keyword_match"]]
print(f"Keyword filter: {len(relevant)}/{len(papers_df)} papers relevant")

# If too many, use DeepSeek to rank
if len(relevant) > 50:
    print(f"Too many papers ({len(relevant)}), using DeepSeek to select top 50 most relevant...")
    abstract_batch = "\n\n".join([
        f"[{i}] {r['arxiv_id']}: {r['title'][:100]}" 
        for i, r in relevant.head(80).iterrows()
    ])
    
    prompt = f"""Rate each paper below by relevance to "Chain-of-Thought mathematical reasoning in Large Language Models" on a scale 1-5.
Return ONLY the paper indices and scores, one per line: [index] score

Papers:
{abstract_batch}"""
    
    response = client.complete(prompt, temperature=0.0, max_tokens=500)
    if response:
        relevant_ids = set()
        for line in response.strip().split("\n"):
            parts = line.strip("[]").split("]")
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    score = int(parts[1].strip())
                    if score >= 4:
                        relevant_ids.add(idx)
                except:
                    pass
        if relevant_ids:
            relevant = relevant.loc[list(relevant_ids & set(relevant.index))]
            print(f"DeepSeek selected {len(relevant)} highly relevant papers")

# Limit to manageable amount
papers_to_process = relevant.head(30)
print(f"Processing {len(papers_to_process)} papers")

# ============================================================
# STEP 2: Load paper text
# ============================================================
print("\n--- STEP 2: Loading paper text ---")
text_df = pd.read_csv(os.path.join(OUTPUT_DIR, "papers_with_text.csv"))
papers_to_process = papers_to_process.merge(
    text_df[["arxiv_id", "full_text"]], on="arxiv_id", how="inner"
)
papers_to_process = papers_to_process[papers_to_process["full_text"].str.len() > 100]
print(f"Papers with text: {len(papers_to_process)}")

# ============================================================
# STEP 3: Extract claims via DeepSeek API
# ============================================================
print("\n--- STEP 3: Extracting claims ---")
extractor = ClaimExtractor(api_client=client)
cleaner = DataCleaner(ontology=ONTOLOGY)

all_claims = []
batch_save = 5
t0 = time.time()

for idx, (_, row) in enumerate(papers_to_process.iterrows()):
    arxiv_id = row["arxiv_id"]
    text = str(row.get("full_text", ""))[:12000]
    title = str(row.get("title", ""))[:80]
    
    print(f"  [{idx+1}/{len(papers_to_process)}] {arxiv_id}: {title}...", end=" ")
    
    try:
        claims = extractor.extract_from_text(text, ONTOLOGY.concepts, ONTOLOGY.relations)
        for c in claims:
            c["arxiv_id"] = arxiv_id
            c["title"] = title
            c["citations"] = row.get("citations", 0)
        all_claims.extend(claims)
        print(f"{len(claims)} claims")
    except Exception as e:
        print(f"ERROR: {e}")
    
    if (idx + 1) % batch_save == 0:
        pd.DataFrame(all_claims).to_csv(os.path.join(OUTPUT_DIR, "all_claims_raw.csv"), index=False)
        print(f"    [saved {len(all_claims)} total]")

elapsed = time.time() - t0
print(f"\nExtraction: {len(all_claims)} claims from {len(papers_to_process)} papers ({elapsed:.0f}s, {elapsed/len(papers_to_process):.1f}s/paper)")

# Save
raw_df = pd.DataFrame(all_claims)
raw_df.to_csv(os.path.join(OUTPUT_DIR, "all_claims_raw.csv"), index=False)

# Clean
print("\n--- STEP 4: Cleaning claims ---")
cleaned_df = cleaner.process(raw_df)
cleaned_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_claims.csv"), index=False)
report = cleaner.generate_cleaning_report(raw_df, cleaned_df)
print(f"Cleaned: {len(cleaned_df)} claims (retention: {report['retention_rate']:.1%})")

# ============================================================
# STEP 5: Contradiction Detection
# ============================================================
print("\n--- STEP 5: Detecting contradictions ---")
detector = ContradictionDetector(
    citation_weight=0.6,
    confidence_weight=0.4,
    mock_mode=False,
    api_client=client
)

candidates = detector.generate_candidate_pairs(cleaned_df)
print(f"Candidate pairs: {len(candidates)}")
contradictions_count = 0

if len(candidates) == 0:
    print("\nNo candidate pairs generated!")
    print("Need more claims about the same (subject, relation, object).")
    print("\nTop claim groups:")
    groups = cleaned_df.groupby(["subject", "relation", "object"]).size().sort_values(ascending=False)
    for (s, r, o), count in groups.head(15).items():
        print(f"  {count}x: {s} {r} {o}")
else:
    # Classify
    if len(candidates) <= 20:
        print(f"Classifying {len(candidates)} pairs with DeepSeek API...")
        classified = detector.classify_all_pairs(candidates)
    else:
        print(f"Many pairs ({len(candidates)}), applying rule engine first...")
        from rule_engine import RuleEngine
        engine = RuleEngine()
        rule_results = engine.process_dataframe(candidates)
        low_conf = rule_results[rule_results["rule_confidence"] <= 0.3]
        print(f"  Rule engine: {len(candidates) - len(low_conf)} classified, API needed: {len(low_conf)}")
        if len(low_conf) > 0:
            api_results = detector.classify_all_pairs(low_conf)
            rule_results.loc[low_conf.index, "contradiction_type"] = api_results["contradiction_type"]
            rule_results.loc[low_conf.index, "classification_reason"] = api_results["classification_reason"]
        classified = rule_results
    
    # Score and filter
    scored = detector.calculate_significance(classified)
    contradictions = detector.filter_contradictions(scored, min_significance=0.0)
    contradictions.to_csv(os.path.join(OUTPUT_DIR, "final_contradictions.csv"), index=False)
    contradictions_count = len(contradictions)
    print(f"Contradictions found: {contradictions_count}")
    
    if len(contradictions) > 0:
        print("\nTop contradictions:")
        for _, row in contradictions.head(5).iterrows():
            print(f"  [{row['contradiction_type']}] sig={row['significance_score']:.1f}")
            print(f"    {row['claim1_text'][:80]}")
            print(f"    {row['claim2_text'][:80]}")
        
        manual_verification_sample(
            contradictions,
            n_samples=min(100, len(contradictions)),
            output_path=os.path.join(OUTPUT_DIR, "manual_verification.csv")
        )
    
    # Rule engine
    engine = RuleEngine()
    cand_path = os.path.join(OUTPUT_DIR, "candidate_contradictions.csv")
    if os.path.exists(cand_path):
        candidates_df = pd.read_csv(cand_path)
        _ = replace_llm_with_rules(candidates_df, engine)
        print(f"Rule engine applied to {len(candidates_df)} candidates")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"  Papers processed: {len(papers_to_process)}")
print(f"  Claims extracted: {len(all_claims)}")
print(f"  Claims cleaned: {len(cleaned_df)}")
print(f"  Candidate pairs: {len(candidates)}")
print(f"  Contradictions: {contradictions_count}")
print(f"  Output: data/final_contradictions.csv")
print("=" * 60)
