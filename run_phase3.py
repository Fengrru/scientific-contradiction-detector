"""Run Phase 3: Contradiction detection on cleaned claims."""
import sys
sys.path.insert(0, "src")

import pandas as pd
import os
from contradiction_detector import ContradictionDetector, manual_verification_sample
from rule_engine import RuleEngine, replace_llm_with_rules
from llm_client import LLMClient

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OUTPUT_DIR = "data"

print("=" * 60)
print("PHASE 3: Contradiction Detection")
print("=" * 60)

# Load cleaned claims
claims_path = os.path.join(OUTPUT_DIR, "cleaned_claims.csv")
if not os.path.exists(claims_path):
    print(f"ERROR: {claims_path} not found!")
    sys.exit(1)

claims_df = pd.read_csv(claims_path)
print(f"Loaded {len(claims_df)} cleaned claims")

# Initialize detector with API
client = LLMClient(api_key=API_KEY, model="deepseek-chat")
detector = ContradictionDetector(
    citation_weight=0.6,
    confidence_weight=0.4,
    mock_mode=False,
    api_client=client
)

# Step 1: Generate candidate pairs
print("\n--- Step 1: Generating candidate pairs ---")
candidates = detector.generate_candidate_pairs(claims_df)
print(f"Generated {len(candidates)} candidate pairs")
candidates_path = os.path.join(OUTPUT_DIR, "candidate_contradictions.csv")
candidates.to_csv(candidates_path, index=False)
print(f"Saved to {candidates_path}")

if len(candidates) == 0:
    print("No candidate pairs found. Need more claims with same (subject, relation, object) groups.")
    print("\nChecking claim groups:")
    grouped = claims_df.groupby(["subject", "relation", "object"]).size().sort_values(ascending=False)
    print(grouped.head(10))
    sys.exit(0)

# Step 2: Classify contradictions
print(f"\n--- Step 2: Classifying {len(candidates)} candidate pairs ---")
if len(candidates) > 30:
    print(f"Many pairs ({len(candidates)}), applying rule engine first...")
    engine = RuleEngine()
    rule_results = engine.process_dataframe(candidates)
    low_conf = rule_results[rule_results["rule_confidence"] <= 0.3]
    print(f"  Rule engine classifies: {len(candidates) - len(low_conf)} pairs")
    print(f"  API needed for: {len(low_conf)} low-confidence pairs")
    if len(low_conf) > 0:
        api_results = detector.classify_all_pairs(low_conf)
        rule_results.loc[low_conf.index, "contradiction_type"] = api_results["contradiction_type"]
        rule_results.loc[low_conf.index, "classification_reason"] = api_results["classification_reason"]
    classified = rule_results
else:
    classified = detector.classify_all_pairs(candidates)

# Step 3: Calculate significance
print("\n--- Step 3: Calculating significance ---")
scored = detector.calculate_significance(classified)

# Step 4: Filter to actual contradictions
print("\n--- Step 4: Filtering contradictions ---")
contradictions = detector.filter_contradictions(scored, min_significance=0.0)

output_path = os.path.join(OUTPUT_DIR, "final_contradictions.csv")
contradictions.to_csv(output_path, index=False)

print(f"\n{'=' * 60}")
print(f"PHASE 3 COMPLETE")
print(f"  Candidates: {len(candidates)}")
print(f"  Contradictions found: {len(contradictions)}")
print(f"  Saved to: {output_path}")
print(f"{'=' * 60}")

# Show top results
if len(contradictions) > 0:
    print("\nTop contradictions:")
    for idx, row in contradictions.head(10).iterrows():
        print(f"  [{row['contradiction_type']}] sig={row['significance_score']:.1f}")
        print(f"    C1: {row['claim1_text']}")
        print(f"    C2: {row['claim2_text']}")

    # Manual verification sample
    verification_df = manual_verification_sample(
        contradictions,
        n_samples=min(100, len(contradictions)),
        output_path=os.path.join(OUTPUT_DIR, "manual_verification.csv")
    )
    print(f"\nPrepared {len(verification_df)} contradictions for manual verification")

# Rule engine
print("\n--- Rule Engine ---")
engine = RuleEngine()
summary = engine.get_rule_summary()
print(f"Loaded {summary['total_rules']} rules")

# Apply rules to update candidates
if os.path.exists(candidates_path):
    candidates_df = pd.read_csv(candidates_path)
    hybrid_results = replace_llm_with_rules(candidates_df, engine)
    hybrid_results.to_csv(candidates_path, index=False)
    print(f"Applied rule engine to {len(hybrid_results)} candidates")
