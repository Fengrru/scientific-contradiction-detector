"""Run Phase 4: Paper generation."""
import sys
sys.path.insert(0, "src")

import pandas as pd
import os
from paper_generator import PaperGenerator, generate_paper_summary

print("=" * 60)
print("PHASE 4: Paper Generation")
print("=" * 60)

# Load data
claims_df = pd.read_csv("data/cleaned_claims.csv") if os.path.exists("data/cleaned_claims.csv") else pd.DataFrame()
contradictions_df = pd.read_csv("data/final_contradictions.csv") if os.path.exists("data/final_contradictions.csv") else pd.DataFrame()
papers_df = pd.read_csv("data/relevant_core_papers.csv") if os.path.exists("data/relevant_core_papers.csv") else (
    pd.read_csv("data/core_papers.csv") if os.path.exists("data/core_papers.csv") else pd.DataFrame()
)

print(f"Claims: {len(claims_df)}")
print(f"Contradictions: {len(contradictions_df)}")
print(f"Papers: {len(papers_df)}")

# Generate paper
generator = PaperGenerator(
    domain="Chain-of-Thought Mathematical Reasoning in Large Language Models",
    target_venue="arXiv"
)

paper_text = generator.generate_full_paper(contradictions_df, claims_df, papers_df)

# Save as .tex
paper_path = "papers/contradiction_paper.tex"
os.makedirs(os.path.dirname(paper_path), exist_ok=True)
with open(paper_path, "w", encoding="utf-8") as f:
    f.write(paper_text)
print(f"\nPaper saved to {paper_path}")

# Generate README summary
summary = generate_paper_summary(contradictions_df)
readme_path = "results/README_SUMMARY.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(summary)
print(f"Summary saved to {readme_path}")

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
