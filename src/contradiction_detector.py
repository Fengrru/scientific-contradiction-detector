"""
Contradiction detection engine.
Generates candidate pairs, classifies contradictions, and calculates significance.

Author: AI Scientist
Date: 2026-04-20
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import logging
import re
from tqdm import tqdm
import torch

from claim_extractor import ClaimExtractor
from ontology import ONTOLOGY
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContradictionDetector:
    """Detect contradictions between scientific claims."""
    
    def __init__(self,
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 use_4bit: bool = True,
                 citation_weight: float = 0.6,
                 confidence_weight: float = 0.4,
                 mock_mode: bool = False,
                 api_client: Optional[LLMClient] = None):
        """
        Initialize contradiction detector.
        
        Args:
            model_name: LLM for contradiction classification
            use_4bit: Use 4-bit quantization
            citation_weight: Weight for citation score in significance
            confidence_weight: Weight for confidence score in significance
            mock_mode: Run in mock mode (skip classifier initialization)
            api_client: LLMClient for API-based classification
        """
        self.citation_weight = citation_weight
        self.confidence_weight = confidence_weight
        self.mock_mode = mock_mode
        self.api_client = api_client
        self.api_mode = api_client is not None and api_client.available
        
        # Initialize classifier
        if mock_mode:
            logger.info("Mock mode: skipping classifier initialization")
            self.classifier = None
        elif self.api_mode:
            logger.info(f"API mode enabled with {api_client.model}")
            self.classifier = None
        else:
            logger.info("Initializing contradiction classifier...")
            self.classifier = ClaimExtractor(
                model_name=model_name,
                use_4bit=use_4bit,
                temperature=0.0  # Deterministic for classification
            )
    
    # Opposite relation pairs for Scientific_Dispute detection
    OPPOSITE_RELATIONS = {
        ("improves", "degrades"), ("degrades", "improves"),
        ("increases", "reduces"), ("reduces", "increases"),
        ("outperforms", "degrades"), ("degrades", "outperforms"),
        ("enables", "degrades"), ("degrades", "enables"),
        ("correlates_with", "contradicts"), ("contradicts", "correlates_with"),
    }
    
    def generate_candidate_pairs(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate candidate contradiction pairs.
        Strategy 1: Same (subject, relation, object) → experimental condition contradictions
        Strategy 2: Same (subject, object) but OPPOSITE relations → scientific disputes
        
        Args:
            claims_df: DataFrame with cleaned claims
            
        Returns:
            DataFrame with candidate pairs
        """
        logger.info("Generating candidate contradiction pairs...")
        
        candidate_pairs = []
        seen_pairs = set()  # prevent duplicates
        
        def make_pair(claim1, claim2, group_key, strategy):
            pair_id = tuple(sorted([str(claim1.name), str(claim2.name)]))
            if pair_id in seen_pairs:
                return None
            seen_pairs.add(pair_id)
            
            return {
                "claim1_id": claim1.name,
                "claim1_arxiv": claim1["arxiv_id"],
                "claim1_title": claim1.get("title", ""),
                "claim1_text": f"{claim1['subject']} {claim1['relation']} {claim1['object']}",
                "claim1_condition": claim1["condition"],
                "claim1_citations": claim1.get("citations", 0),
                "claim1_confidence": int(claim1["confidence"]),
                "claim2_id": claim2.name,
                "claim2_arxiv": claim2["arxiv_id"],
                "claim2_title": claim2.get("title", ""),
                "claim2_text": f"{claim2['subject']} {claim2['relation']} {claim2['object']}",
                "claim2_condition": claim2["condition"],
                "claim2_citations": claim2.get("citations", 0),
                "claim2_confidence": int(claim2["confidence"]),
                "group_key": group_key,
                "strategy": strategy
            }
        
        # Strategy 1: Same (subject, relation, object) → experimental contradictions
        grouped_sro = claims_df.groupby(["subject", "relation", "object"])
        
        for (subject, relation, obj), group in tqdm(grouped_sro, desc="Strategy 1 (same SRO)"):
            if len(group) < 2:
                continue
            group = group.reset_index(drop=True)
            for i, j in combinations(range(len(group)), 2):
                if group.iloc[i]["arxiv_id"] == group.iloc[j]["arxiv_id"]:
                    continue
                pair = make_pair(group.iloc[i], group.iloc[j],
                               f"{subject}|{relation}|{obj}", "same_SRO")
                if pair:
                    candidate_pairs.append(pair)
        
        # Strategy 2: Same (subject, object) BUT opposite relations → disputes
        grouped_so = claims_df.groupby(["subject", "object"])
        
        for (subject, obj), group in tqdm(grouped_so, desc="Strategy 2 (opposite rels)"):
            if len(group) < 2:
                continue
            group = group.reset_index(drop=True)
            for i, j in combinations(range(len(group)), 2):
                c1, c2 = group.iloc[i], group.iloc[j]
                if c1["arxiv_id"] == c2["arxiv_id"]:
                    continue
                rel_pair = (c1["relation"], c2["relation"])
                if rel_pair in self.OPPOSITE_RELATIONS:
                    pair = make_pair(c1, c2,
                                   f"{subject}|OPPOSITE|{obj}", "opposite_rels")
                    if pair:
                        candidate_pairs.append(pair)
        
        candidates_df = pd.DataFrame(candidate_pairs)
        modes = candidates_df.get("strategy", pd.Series()).value_counts().to_dict() if len(candidates_df) > 0 else {}
        logger.info(f"Generated {len(candidates_df)} candidate pairs: {modes}")
        
        return candidates_df
    
    def _build_classification_prompt(self,
                                     claim1_text: str,
                                     claim1_condition: str,
                                     claim2_text: str,
                                     claim2_condition: str) -> str:
        """
        Build prompt for contradiction classification.
        
        Args:
            claim1_text: First claim description
            claim1_condition: First claim conditions
            claim2_text: Second claim description
            claim2_condition: Second claim conditions
            
        Returns:
            Classification prompt
        """
        contradiction_types = "\n".join([f"{i+1}. {ct}" for i, ct in enumerate(ONTOLOGY.contradiction_types)])
        
        prompt = f"""Analyze these two claims from the scientific literature and identify what kind of relationship they have.

CLAIM 1:
Statement: {claim1_text}
Conditions: {claim1_condition}

CLAIM 2:
Statement: {claim2_text}
Conditions: {claim2_condition}

CATEGORIES (pick the ONE that best fits):
4 = Scientific_Dispute: Claims DIRECTLY CONTRADICT — one says X helps, the other says X hurts. Or they use OPPOSITE relations (improves vs degrades, outperforms vs degrades).
1 = Experimental_Condition_Contradiction: Claims appear to differ but ONLY because they use different experimental setups (different datasets, model sizes, prompts, temperatures).
2 = Measurement_Method_Contradiction: Claims differ because they measure results differently (accuracy vs F1, human eval vs automatic, different benchmarks).
3 = Statistical_Error_Contradiction: Claims differ due to small sample sizes, overlapping confidence intervals, or non-significant p-values.
5 = Logical_Contradiction: Claims are logically incompatible — one claim mathematically implies the negation of the other.
6 = No_Contradiction: The two claims are about DIFFERENT things, or they actually AGREE with each other.

RULES:
- FIRST check for Scientific_Dispute (opposite relations is the strongest signal)
- THEN check if different conditions explain the difference
- Only pick type 1 if the claims would AGREE under the SAME conditions
- If the statements talk about DIFFERENT subjects/objects, pick type 6

OUTPUT:
Type: [number 1-6]
Reason: [one sentence explanation]"""
        
        return prompt
    
    def classify_contradiction(self,
                            claim1_text: str,
                            claim1_condition: str,
                            claim2_text: str,
                            claim2_condition: str) -> Tuple[str, str]:
        """
        Classify contradiction type for a pair.
        
        Args:
            claim1_text: First claim
            claim1_condition: First conditions
            claim2_text: Second claim
            claim2_condition: Second conditions
            
        Returns:
            (contradiction_type, reason)
        """
        # Mock mode: skip classification, mark as unknown
        if self.mock_mode:
            logger.warning("Mock mode: returning No_Contradiction for all pairs")
            return "No_Contradiction", "Mock mode — classification skipped"
        
        prompt = self._build_classification_prompt(
            claim1_text, claim1_condition,
            claim2_text, claim2_condition
        )
        
        # API mode: use DeepSeek API
        if self.api_mode:
            response = self.api_client.complete(
                prompt,
                temperature=0.0,
                max_tokens=100
            )
            if response is None:
                return "No_Contradiction", "API call failed"
        else:
            # Local model mode
            inputs = self.classifier.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.classifier.max_length
            ).to(self.classifier.model.device)
            
            with torch.no_grad():
                outputs = self.classifier.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.classifier.tokenizer.eos_token_id
                )
            
            response = self.classifier.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        type_match = re.search(r'Type:\s*(\d)', response)
        if type_match:
            type_num = int(type_match.group(1))
            if 1 <= type_num <= 6:
                contradiction_type = ONTOLOGY.contradiction_types[type_num - 1]
            else:
                contradiction_type = "No_Contradiction"
        else:
            contradiction_type = "No_Contradiction"
        
        # Extract reason
        reason_match = re.search(r'Reason:\s*(.+)', response)
        reason = reason_match.group(1) if reason_match else ""
        
        return contradiction_type, reason
    
    def classify_all_pairs(self, candidates_df: pd.DataFrame, rule_engine=None) -> pd.DataFrame:
        """
        Classify all candidate pairs. Uses rule engine first, API as fallback.
        
        Args:
            candidates_df: DataFrame with candidate pairs
            rule_engine: Optional RuleEngine for hybrid classification
            
        Returns:
            DataFrame with contradiction types
        """
        logger.info(f"Classifying {len(candidates_df)} candidate pairs...")
        
        types = []
        reasons = []
        
        for idx, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Classifying"):
            try:
                # Step 1: Try rule engine first
                if rule_engine is not None:
                    claim1_dict = {
                        "subject": row.get("claim1_text", "").split()[0] if row.get("claim1_text") else "",
                        "relation": row.get("claim1_text", "").split()[1] if len(row.get("claim1_text", "").split()) > 1 else "",
                        "object": " ".join(row.get("claim1_text", "").split()[2:]) if len(row.get("claim1_text", "").split()) > 2 else "",
                        "condition": row.get("claim1_condition", "")
                    }
                    claim2_dict = {
                        "subject": row.get("claim2_text", "").split()[0] if row.get("claim2_text") else "",
                        "relation": row.get("claim2_text", "").split()[1] if len(row.get("claim2_text", "").split()) > 1 else "",
                        "object": " ".join(row.get("claim2_text", "").split()[2:]) if len(row.get("claim2_text", "").split()) > 2 else "",
                        "condition": row.get("claim2_condition", "")
                    }
                    rule_type, rule_conf, _ = rule_engine.analyze_pair(claim1_dict, claim2_dict)
                    
                    # Use rule engine result if confident
                    if rule_type != "No_Contradiction" and rule_conf >= 0.2:
                        types.append(rule_type)
                        reasons.append(f"Rule engine: {rule_type} (confidence={rule_conf:.2f})")
                        continue
                
                # Step 2: Fall back to API/LLM
                ctype, reason = self.classify_contradiction(
                    row["claim1_text"],
                    row["claim1_condition"],
                    row["claim2_text"],
                    row["claim2_condition"]
                )
                types.append(ctype)
                reasons.append(reason)
            except Exception as e:
                logger.warning(f"Classification error for pair {idx}: {e}")
                types.append("No_Contradiction")
                reasons.append("")
        
        candidates_df["contradiction_type"] = types
        candidates_df["classification_reason"] = reasons
        
        return candidates_df
    
    def calculate_significance(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate significance score for each contradiction.
        
        Significance = citation_weight * log_avg_citations + confidence_weight * avg_confidence
        Uses log1p to prevent single high-citation paper from dominating.
        
        Args:
            candidates_df: DataFrame with classified pairs
            
        Returns:
            DataFrame with significance scores
        """
        df = candidates_df.copy()
        
        # Calculate average citation and confidence
        df["avg_citations"] = (df["claim1_citations"].astype(float) + df["claim2_citations"].astype(float)) / 2
        df["avg_confidence"] = (df["claim1_confidence"].astype(float) + df["claim2_confidence"].astype(float)) / 2
        
        # Apply log scaling to citations to prevent dominance by single high-citation paper
        df["log_citations"] = np.log1p(df["avg_citations"])
        
        # Calculate significance score with log-scaled citations
        df["significance_score"] = (
            self.citation_weight * df["log_citations"] * 10 +  # Scale factor for readability
            self.confidence_weight * df["avg_confidence"] * 10
        )
        
        # Sort by significance
        df = df.sort_values("significance_score", ascending=False).reset_index(drop=True)
        
        return df
    
    def filter_contradictions(self,
                             candidates_df: pd.DataFrame,
                              min_significance: float = 5.0) -> pd.DataFrame:
        """
        Filter to actual contradictions (not No_Contradiction) above threshold.
        
        Args:
            candidates_df: DataFrame with classified pairs
            min_significance: Minimum significance score
            
        Returns:
            DataFrame with filtered contradictions
        """
        # Filter out "No_Contradiction"
        contradictions = candidates_df[
            candidates_df["contradiction_type"] != "No_Contradiction"
        ].copy()
        
        # Filter by significance
        contradictions = contradictions[
            contradictions["significance_score"] >= min_significance
        ].copy()
        
        logger.info(f"Found {len(contradictions)} significant contradictions")
        
        return contradictions
    
    def run(self, claims_df: pd.DataFrame, rule_engine=None) -> pd.DataFrame:
        """
        Execute full contradiction detection pipeline.
        
        Args:
            claims_df: Cleaned claims DataFrame
            rule_engine: Optional RuleEngine for hybrid classification
            
        Returns:
            DataFrame with detected contradictions
        """
        # Step 1: Generate candidates
        candidates = self.generate_candidate_pairs(claims_df)
        
        # Step 2: Classify (rule engine + API hybrid)
        classified = self.classify_all_pairs(candidates, rule_engine=rule_engine)
        
        # Step 3: Calculate significance
        scored = self.calculate_significance(classified)
        
        # Step 4: Filter
        contradictions = self.filter_contradictions(scored)
        
        return contradictions


def manual_verification_sample(contradictions_df: pd.DataFrame,
                               n_samples: int = 100,
                               output_path: str = "./data/manual_verification.csv") -> pd.DataFrame:
    """
    Prepare top N contradictions for manual verification.
    
    Args:
        contradictions_df: DataFrame with contradictions
        n_samples: Number of samples to verify
        output_path: Path to save verification sheet
        
    Returns:
        DataFrame ready for manual verification
    """
    # Take top N by significance
    top_n = contradictions_df.head(n_samples).copy()
    
    # Add verification columns
    top_n["manual_verified"] = False
    top_n["is_true_positive"] = None  # User fills this
    top_n["notes"] = ""  # User notes
    
    # Save
    top_n.to_csv(output_path, index=False)
    
    logger.info(f"Prepared {len(top_n)} contradictions for manual verification: {output_path}")
    
    return top_n


if __name__ == "__main__":
    # Test with sample data
    sample_claims = pd.DataFrame([
        {
            "arxiv_id": "paper1",
            "title": "Paper 1",
            "subject": "Chain-of-Thought",
            "relation": "improves",
            "object": "Arithmetic Accuracy",
            "condition": "GSM8K with GPT-3.5",
            "evidence_type": "Experimental_Result",
            "confidence": 5,
            "citations": 50
        },
        {
            "arxiv_id": "paper2",
            "title": "Paper 2",
            "subject": "Chain-of-Thought",
            "relation": "improves",
            "object": "Arithmetic Accuracy",
            "condition": "GSM8K with Llama-2",
            "evidence_type": "Experimental_Result",
            "confidence": 4,
            "citations": 30
        },
        {
            "arxiv_id": "paper3",
            "title": "Paper 3",
            "subject": "Chain-of-Thought",
            "relation": "degrades",
            "object": "Arithmetic Accuracy",
            "condition": "Complex problems with GPT-4",
            "evidence_type": "Experimental_Result",
            "confidence": 3,
            "citations": 20
        }
    ])
    
    print("Sample claims:")
    print(sample_claims)
    
    # Test candidate generation
    detector = ContradictionDetector()
    candidates = detector.generate_candidate_pairs(sample_claims)
    
    print(f"\nGenerated {len(candidates)} candidate pairs:")
    print(candidates[["claim1_arxiv", "claim2_arxiv", "claim1_text", "group_key"]])
