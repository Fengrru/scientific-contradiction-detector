"""
Rule engine construction from verified contradictions.
Derives logical rules from manually verified contradictions to replace LLM classification.

Author: AI Scientist
Date: 2026-04-20
"""

import pandas as pd
from typing import List, Dict, Tuple, Callable
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContradictionRule:
    """A single contradiction detection rule."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 condition_func: Callable,
                 contradiction_type: str,
                 confidence_boost: float = 0.0):
        """
        Initialize a rule.
        
        Args:
            name: Rule identifier
            description: Human-readable description
            condition_func: Function that takes (claim1, claim2) and returns bool
            contradiction_type: Type of contradiction this rule detects
            confidence_boost: Score boost when rule matches
        """
        self.name = name
        self.description = description
        self.condition_func = condition_func
        self.contradiction_type = contradiction_type
        self.confidence_boost = confidence_boost
    
    def applies_to(self, claim1: Dict, claim2: Dict) -> bool:
        """Check if rule applies to claim pair."""
        try:
            return self.condition_func(claim1, claim2)
        except:
            return False


class RuleEngine:
    """Engine for rule-based contradiction detection."""
    
    def __init__(self):
        """Initialize rule engine."""
        self.rules: List[ContradictionRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default contradiction detection rules."""
        
        # Rule 1: Opposite relations on same subject-object pair
        def opposite_relations(c1, c2):
            opposite_pairs = [
                ("improves", "degrades"),
                ("increases", "reduces"),
                ("outperforms", "degrades"),
            ]
            if c1["subject"] == c2["subject"] and c1["object"] == c2["object"]:
                return (c1["relation"], c2["relation"]) in opposite_pairs or \
                       (c2["relation"], c1["relation"]) in opposite_pairs
            return False
        
        self.add_rule(ContradictionRule(
            name="opposite_relations",
            description="Same subject-object pair with opposite relations",
            condition_func=opposite_relations,
            contradiction_type="Scientific_Dispute",
            confidence_boost=0.3
        ))
        
        # Rule 2: Different datasets with different conclusions
        def different_datasets_conflict(c1, c2):
            if c1["subject"] == c2["subject"] and c1["relation"] == c2["relation"]:
                # Check if conditions mention different datasets
                c1_has_dataset = any(x in c1.get("condition", "").lower() 
                                   for x in ["gsm8k", "math", "svamp", "dataset"])
                c2_has_dataset = any(x in c2.get("condition", "").lower() 
                                   for x in ["gsm8k", "math", "svamp", "dataset"])
                if c1_has_dataset and c2_has_dataset:
                    # Different conditions but same conclusion type
                    return c1.get("condition", "") != c2.get("condition", "")
            return False
        
        self.add_rule(ContradictionRule(
            name="experimental_condition_conflict",
            description="Same claim but different experimental conditions",
            condition_func=different_datasets_conflict,
            contradiction_type="Experimental_Condition_Contradiction",
            confidence_boost=0.2
        ))
        
        # Rule 3: Different models with opposite performance claims
        def model_comparison_conflict(c1, c2):
            model_keywords = ["gpt-4", "gpt-3.5", "llama", "claude", "palm", "codex"]
            
            c1_condition = c1.get("condition", "").lower()
            c2_condition = c2.get("condition", "").lower()
            
            # Check if different models mentioned
            c1_model = [m for m in model_keywords if m in c1_condition]
            c2_model = [m for m in model_keywords if m in c2_condition]
            
            if c1_model and c2_model and c1_model[0] != c2_model[0]:
                # Same subject-relation-object but different models
                if (c1["subject"] == c2["subject"] and 
                    c1["relation"] == c2["relation"] and
                    c1["object"] == c2["object"]):
                    return True
            return False
        
        self.add_rule(ContradictionRule(
            name="model_comparison_conflict",
            description="Different LLM models show different results for same claim",
            condition_func=model_comparison_conflict,
            contradiction_type="Experimental_Condition_Contradiction",
            confidence_boost=0.25
        ))
        
        # Rule 4: Statistical magnitude disagreement
        def statistical_disagreement(c1, c2):
            # Extract numbers from object field
            c1_nums = re.findall(r'\d+\.?\d*', str(c1.get("object", "")))
            c2_nums = re.findall(r'\d+\.?\d*', str(c2.get("object", "")))
            
            if c1_nums and c2_nums:
                try:
                    n1 = float(c1_nums[0])
                    n2 = float(c2_nums[0])
                    
                    # Same subject-relation, different magnitudes
                    if (c1["subject"] == c2["subject"] and 
                        c1["relation"] == c2["relation"] and
                        c1["object"] != c2["object"]):
                        
                        # Significant difference (>10% relative)
                        if n1 > 0 and abs(n1 - n2) / n1 > 0.1:
                            return True
                except:
                    pass
            return False
        
        self.add_rule(ContradictionRule(
            name="statistical_disagreement",
            description="Numerical values differ significantly for same measurement",
            condition_func=statistical_disagreement,
            contradiction_type="Statistical_Error_Contradiction",
            confidence_boost=0.2
        ))
    
    def add_rule(self, rule: ContradictionRule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name}")
    
    def analyze_pair(self, claim1: Dict, claim2: Dict) -> Tuple[str, float, List[str]]:
        """
        Analyze a claim pair using all rules.
        
        Args:
            claim1: First claim dictionary
            claim2: Second claim dictionary
            
        Returns:
            (contradiction_type, confidence, matched_rules)
        """
        matched_rules = []
        detected_types = []
        confidence = 0.0
        
        for rule in self.rules:
            if rule.applies_to(claim1, claim2):
                matched_rules.append(rule.name)
                detected_types.append(rule.contradiction_type)
                confidence += rule.confidence_boost
        
        if not matched_rules:
            return "No_Contradiction", 0.0, []
        
        # Take most specific contradiction type
        type_priority = [
            "Scientific_Dispute",
            "Logical_Contradiction",
            "Experimental_Condition_Contradiction", 
            "Measurement_Method_Contradiction",
            "Statistical_Error_Contradiction"
        ]
        
        final_type = "Scientific_Dispute"  # Default
        for t in type_priority:
            if t in detected_types:
                final_type = t
                break
        
        # Cap confidence
        confidence = min(confidence, 1.0)
        
        return final_type, confidence, matched_rules
    
    def process_dataframe(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all candidate pairs in DataFrame.
        
        Args:
            candidates_df: DataFrame with candidate pairs
            
        Returns:
            DataFrame with rule-based classifications
        """
        df = candidates_df.copy()
        
        types = []
        confidences = []
        rules_matched = []
        
        for idx, row in df.iterrows():
            # Reconstruct claim dictionaries
            claim1 = {
                "subject": row.get("claim1_text", "").split()[0] if row.get("claim1_text") else "",
                "relation": row.get("claim1_text", "").split()[1] if len(row.get("claim1_text", "").split()) > 1 else "",
                "object": " ".join(row.get("claim1_text", "").split()[2:]) if len(row.get("claim1_text", "").split()) > 2 else "",
                "condition": row.get("claim1_condition", "")
            }
            
            claim2 = {
                "subject": row.get("claim2_text", "").split()[0] if row.get("claim2_text") else "",
                "relation": row.get("claim2_text", "").split()[1] if len(row.get("claim2_text", "").split()) > 1 else "",
                "object": " ".join(row.get("claim2_text", "").split()[2:]) if len(row.get("claim2_text", "").split()) > 2 else "",
                "condition": row.get("claim2_condition", "")
            }
            
            ctype, conf, rules = self.analyze_pair(claim1, claim2)
            
            types.append(ctype)
            confidences.append(conf)
            rules_matched.append(",".join(rules))
        
        df["rule_based_type"] = types
        df["rule_confidence"] = confidences
        df["matched_rules"] = rules_matched
        
        return df
    
    def derive_rules_from_verified(self,
                                   verified_df: pd.DataFrame,
                                   min_support: int = 3) -> List[ContradictionRule]:
        """
        Derive new rules from manually verified contradictions.
        
        Args:
            verified_df: DataFrame with manual verification results
            min_support: Minimum occurrences to create a rule
            
        Returns:
            List of derived rules
        """
        # Filter to true positives
        true_positives = verified_df[verified_df.get("is_true_positive", False) == True]
        
        logger.info(f"Analyzing {len(true_positives)} verified contradictions for rule derivation...")
        
        # Group by contradiction type and find patterns
        derived_rules = []
        
        for ctype, group in true_positives.groupby("contradiction_type"):
            if len(group) >= min_support:
                # Analyze common patterns
                common_conditions = self._find_common_patterns(group)
                
                if common_conditions:
                    rule = ContradictionRule(
                        name=f"derived_{ctype.lower()}",
                        description=f"Derived from {len(group)} verified {ctype} cases",
                        condition_func=common_conditions,
                        contradiction_type=ctype,
                        confidence_boost=0.4
                    )
                    derived_rules.append(rule)
                    logger.info(f"Derived rule for {ctype} from {len(group)} cases")
        
        return derived_rules
    
    def _find_common_patterns(self, verified_group: pd.DataFrame) -> Callable:
        """
        Find common patterns in verified contradictions.
        
        Args:
            verified_group: DataFrame of verified contradictions of same type
            
        Returns:
            Condition function capturing the pattern
        """
        # Placeholder: pattern mining from verified contradictions
        # Future: analyze condition fields, subject-object pairs for common patterns
        # and generate new ContradictionRule instances automatically
        
        def pattern_matcher(c1, c2):
            # Not yet implemented — requires sufficient verified samples
            return False
        
        return pattern_matcher
    
    def get_rule_summary(self) -> Dict:
        """Get summary of all rules in engine."""
        return {
            "total_rules": len(self.rules),
            "rules": [
                {
                    "name": r.name,
                    "type": r.contradiction_type,
                    "description": r.description
                }
                for r in self.rules
            ]
        }


def replace_llm_with_rules(candidates_df: pd.DataFrame,
                          rule_engine: RuleEngine) -> pd.DataFrame:
    """
    Replace LLM classifications with rule-based where confident.
    
    Args:
        candidates_df: DataFrame with LLM classifications
        rule_engine: Rule engine instance
        
    Returns:
        DataFrame with hybrid classification
    """
    df = candidates_df.copy()
    
    # Add rule-based classifications
    df = rule_engine.process_dataframe(df)
    
    # Use rule-based where confident, otherwise keep LLM
    df["final_type"] = df.apply(
        lambda row: row["rule_based_type"] 
        if row["rule_confidence"] > 0.3 
        else row.get("contradiction_type", "No_Contradiction"),
        axis=1
    )
    
    df["final_confidence"] = df["rule_confidence"]
    
    return df


if __name__ == "__main__":
    # Test rule engine
    engine = RuleEngine()
    
    print("Rule Engine Summary:")
    summary = engine.get_rule_summary()
    for rule in summary["rules"]:
        print(f"  - {rule['name']}: {rule['description']}")
    
    # Test with sample claims
    claim1 = {
        "subject": "Chain-of-Thought",
        "relation": "improves",
        "object": "Arithmetic Accuracy",
        "condition": "GSM8K with GPT-3.5"
    }
    
    claim2 = {
        "subject": "Chain-of-Thought",
        "relation": "degrades",
        "object": "Arithmetic Accuracy",
        "condition": "Complex problems with GPT-4"
    }
    
    ctype, conf, rules = engine.analyze_pair(claim1, claim2)
    print(f"\nTest pair analysis:")
    print(f"  Type: {ctype}")
    print(f"  Confidence: {conf}")
    print(f"  Matched rules: {rules}")
