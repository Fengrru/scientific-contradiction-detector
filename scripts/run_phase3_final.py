"""Quick Phase 3: finish classification + generate paper."""
import sys, os
sys.path.insert(0, "../src")
import pandas as pd
from contradiction_detector import ContradictionDetector, manual_verification_sample
from rule_engine import RuleEngine, replace_llm_with_rules
from paper_generator import PaperGenerator, generate_paper_summary
from llm_client import LLMClient
from ontology import ONTOLOGY

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OUTPUT_DIR = "data"

# Load candidates (saved by pipeline)
candidates = pd.read_csv(os.path.join(OUTPUT_DIR, "candidate_contradictions.csv"))
print(f"Loaded {len(candidates)} candidates")

if len(candidates) == 0:
    print("No candidates - check cleaned_claims.csv groups")
    cleaned = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_claims.csv"))
    groups = cleaned.groupby(["subject","relation","object"]).size().sort_values(ascending=False)
    print(groups.head(20))
    sys.exit(1)

# Classify first N using DeepSeek API, rest with rule engine
engine = RuleEngine()
summary = engine.get_rule_summary()
print(f"Rule engine: {summary['total_rules']} rules")

# Quick classify with rule engine (instant) for all
candidates_with_rules = engine.process_dataframe(candidates)
print(f"Rule classifications done. Types so far:")
print(candidates_with_rules["rule_based_type"].value_counts().to_dict())

# Now use LLM for pairs where rule confidence is low
low_conf = candidates_with_rules[candidates_with_rules["rule_confidence"] <= 0.3]
print(f"LLM needed for: {len(low_conf)} / {len(candidates)} pairs")

if len(low_conf) > 0:
    client = LLMClient(api_key=API_KEY, model="deepseek-chat")
    detector = ContradictionDetector(
        citation_weight=0.6, confidence_weight=0.4,
        mock_mode=False, api_client=client
    )
    
    print("Classifying low-confidence pairs with DeepSeek API...")
    llm_results = detector.classify_all_pairs(low_conf)
    
    # Merge LLM results back
    candidates_with_rules.loc[low_conf.index, "contradiction_type"] = llm_results["contradiction_type"]
    candidates_with_rules.loc[low_conf.index, "classification_reason"] = llm_results["classification_reason"]

# Final hybrid type
candidates_with_rules["hybrid_type"] = candidates_with_rules.apply(
    lambda r: r["rule_based_type"] if r["rule_confidence"] > 0.3 else r.get("contradiction_type", "No_Contradiction"),
    axis=1
)

# Score and filter
detector2 = ContradictionDetector(citation_weight=0.6, confidence_weight=0.4)
scored = detector2.calculate_significance(candidates_with_rules)
scored["contradiction_type"] = scored["hybrid_type"]
contradictions = detector2.filter_contradictions(scored, min_significance=0.0)
contradictions.to_csv(os.path.join(OUTPUT_DIR, "final_contradictions.csv"), index=False)

print(f"\nFinal contradictions: {len(contradictions)}")
print("Type distribution:")
for t, c in contradictions["contradiction_type"].value_counts().items():
    print(f"  {t}: {c}")

print("\nTop 10 by significance:")
for _, r in contradictions.head(10).iterrows():
    print(f"  [{r['contradiction_type'][:40]:40s}] sig={r['significance_score']:6.1f}")
    print(f"    {r['claim1_text'][:55]}")
    print(f"    {r['claim2_text'][:55]}")

manual_verification_sample(contradictions, n_samples=min(100, len(contradictions)),
                           output_path=os.path.join(OUTPUT_DIR, "manual_verification.csv"))

# Generate paper
papers_path = os.path.join(OUTPUT_DIR, "relevant_papers_with_text.csv")
if not os.path.exists(papers_path):
    papers_path = os.path.join(OUTPUT_DIR, "papers_with_text.csv")
papers_df = pd.read_csv(papers_path)
cleaned_df = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_claims.csv"))
gen = PaperGenerator(domain="Chain-of-Thought Mathematical Reasoning in LLMs", target_venue="arXiv")
tex = gen.generate_full_paper(contradictions, cleaned_df, papers_df)
os.makedirs("papers", exist_ok=True)
with open("papers/contradiction_paper.tex", "w", encoding="utf-8") as f:
    f.write(tex)
summary = generate_paper_summary(contradictions)
with open("results/README_SUMMARY.md", "w", encoding="utf-8") as f:
    f.write(summary)
print("\nPaper and summary saved.")
