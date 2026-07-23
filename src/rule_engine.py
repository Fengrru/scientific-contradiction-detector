"""
Rule engine construction from verified contradictions.
Derives logical rules from manually verified contradictions to replace LLM classification.

Author: AI Scientist
Date: 2026-04-20
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
import re
import logging
from collections import Counter

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
        
        Analyzes verified true-positive contradictions to discover recurring patterns
        (e.g., specific subject-object pairs, condition keyword combinations)
        and generates ContradictionRule instances that capture those patterns.
        
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
                # Analyze common patterns (may return multiple condition functions)
                conditions = self._find_common_patterns(group)
                
                if not isinstance(conditions, list):
                    conditions = [conditions]
                
                for i, cond_func in enumerate(conditions):
                    if cond_func is not None:
                        suffix = f"_{i+1}" if len(conditions) > 1 else ""
                        rule = ContradictionRule(
                            name=f"derived_{ctype.lower()}{suffix}",
                            description=f"Derived from {len(group)} verified {ctype} cases",
                            condition_func=cond_func,
                            contradiction_type=ctype,
                            confidence_boost=0.4
                        )
                        derived_rules.append(rule)
                        logger.info(f"Derived rule '{rule.name}' for {ctype} from {len(group)} cases")
        
        return derived_rules
    
    def auto_derive_and_add_rules(self,
                                   verified_df: pd.DataFrame,
                                   min_support: int = 3) -> int:
        """
        Derive new rules from verified data and add them to the engine.
        
        Convenience method that combines derive_rules_from_verified and add_rule.
        
        Args:
            verified_df: DataFrame with manual verification results
            min_support: Minimum occurrences to create a rule
            
        Returns:
            Number of new rules added
        """
        new_rules = self.derive_rules_from_verified(verified_df, min_support)
        for rule in new_rules:
            self.add_rule(rule)
        return len(new_rules)
    
    def _find_common_patterns(self, verified_group: pd.DataFrame) -> List[Callable]:
        """
        Mine patterns from verified contradictions and return condition functions.
        
        Analyzes subject-relation-object patterns, condition keywords, and their
        co-occurrence to discover rules beyond the default 4 patterns.
        Can return multiple condition functions for different sub-patterns.
        
        Args:
            verified_group: DataFrame of verified contradictions of same type
            
        Returns:
            List of condition function(s) capturing discovered patterns
        """
        if verified_group.empty:
            return []
        
        # ── Helper: parse "subject relation object" text ──
        def parse_sro(text):
            parts = str(text).split()
            return (
                parts[0] if len(parts) > 0 else '',
                parts[1] if len(parts) > 1 else '',
                ' '.join(parts[2:]) if len(parts) > 2 else ''
            )
        
        # ── Extract SRO from claim texts ──
        c1_sro = verified_group['claim1_text'].dropna().apply(parse_sro)
        c2_sro = verified_group['claim2_text'].dropna().apply(parse_sro)
        
        subjects_1 = [s for s, _, _ in c1_sro]
        rels_1    = [r for _, r, _ in c1_sro]
        objs_1    = [o for _, _, o in c1_sro]
        subjects_2 = [s for s, _, _ in c2_sro]
        rels_2    = [r for _, r, _ in c2_sro]
        objs_2    = [o for _, _, o in c2_sro]
        
        # ── Known keywords for condition field mining ──
        KNOWN_DATASETS = {
            'gsm8k', 'math', 'svamp', 'mgsm', 'aqua', 'asdiv',
            'multiarith', 'addsub', 'singleeq', 'gpqa', 'mmlu',
            'bbh', 'halueval', 'mawps', 'tabmwp', 'finqa',
            'strategyqa', 'date', 'sports', 'saycan', 'penguins',
            'rcc-8', 'aerialvln', 'alfworld', 'webshop', 'hotpotqa',
        }
        KNOWN_MODELS = {
            'gpt-4', 'gpt-3.5', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
            'llama-2', 'llama-3', 'llama-3.1', 'llama-3.2', 'llama-2-7b',
            'llama-2-13b', 'llama-2-70b',
            'claude', 'claude-3', 'claude-3.5',
            'palm', 'palm-2', 'gemini', 'gemini-1.5-pro', 'codex',
            'olmo', 'qwen', 'qwen2', 'qwen2.5', 'mistral', 'mixtral',
            'falcon', 'phi-3', 'deepseek', 'o1-mini', 'o3-mini',
            'lamda', 'minerva', 'bloom', 't5', 'bart',
        }
        
        # ── Build condition keyword profiles ──
        all_conditions = pd.concat([
            verified_group['claim1_condition'].dropna().astype(str),
            verified_group['claim2_condition'].dropna().astype(str)
        ])
        
        # Count keyword occurrences across conditions
        kw_counter = Counter()
        for cond in all_conditions:
            cond_lower = cond.lower()
            tokens = set(re.findall(r'[a-zA-Z][a-zA-Z0-9_.-]+', cond_lower))
            kw_counter.update(t for t in tokens if t in KNOWN_DATASETS | KNOWN_MODELS)
        
        # Keep keywords that appear in at least 2 conditions
        distinctive_kws = {kw for kw, cnt in kw_counter.items() if cnt >= 2}
        
        # ── Detect overarching pattern type ──
        n = len(verified_group)
        
        # How many pairs share the same SRO?
        same_sro_count = sum(
            1 for i in range(n)
            if subjects_1[i] == subjects_2[i]
            and rels_1[i] == rels_2[i]
            and objs_1[i] == objs_2[i]
        )
        
        # How many pairs have opposite relations with same subject-object?
        opp_pairs = {
            ('improves', 'degrades'), ('increases', 'reduces'),
            ('outperforms', 'degrades'), ('enables', 'reduces'),
            ('correlates_with', 'contradicts'),
        }
        opposite_rel_count = sum(
            1 for i in range(n)
            if subjects_1[i] == subjects_2[i] and objs_1[i] == objs_2[i]
            and ((rels_1[i], rels_2[i]) in opp_pairs
                 or (rels_2[i], rels_1[i]) in opp_pairs)
        )
        
        # ── Determine majority subject, relation, object ──
        all_subjects = subjects_1 + subjects_2
        all_relations = rels_1 + rels_2
        all_objects = objs_1 + objs_2
        
        top_subject = Counter(all_subjects).most_common(1)[0][0] if all_subjects else ''
        top_relation = Counter(all_relations).most_common(1)[0][0] if all_relations else ''
        top_object = Counter(all_objects).most_common(1)[0][0] if all_objects else ''
        top_subject_count = Counter(all_subjects).most_common(1)[0][1] if all_subjects else 0
        
        # ── Generate condition functions ──
        conditions = []
        
        # Pattern A: Same SRO + different distinctive keywords → Condition Contradiction
        if same_sro_count >= max(2, n * 0.4) and len(distinctive_kws) >= 2:
            def make_sro_kw_matcher(_subj=top_subject, _rel=top_relation,
                                    _obj=top_object, _kws=distinctive_kws.copy()):
                def matcher(c1, c2):
                    s1 = str(c1.get('subject', ''))
                    s2 = str(c2.get('subject', ''))
                    r1 = str(c1.get('relation', ''))
                    r2 = str(c2.get('relation', ''))
                    o1 = str(c1.get('object', ''))
                    o2 = str(c2.get('object', ''))
                    if s1 == s2 and r1 == r2 and o1 == o2:
                        c1_cond = str(c1.get('condition', '')).lower()
                        c2_cond = str(c2.get('condition', '')).lower()
                        c1_kws = {k for k in _kws if k in c1_cond}
                        c2_kws = {k for k in _kws if k in c2_cond}
                        return bool(c1_kws and c2_kws and c1_kws != c2_kws)
                    return False
                return matcher
            conditions.append(make_sro_kw_matcher())
        
        # Pattern B: Specific subject with opposite relations → Scientific Dispute
        if opposite_rel_count >= max(2, n * 0.3):
            def make_opposite_rel_matcher(_subj=top_subject, _obj=top_object):
                def matcher(c1, c2):
                    s1, s2 = str(c1.get('subject', '')), str(c2.get('subject', ''))
                    r1, r2 = str(c1.get('relation', '')), str(c2.get('relation', ''))
                    o1, o2 = str(c1.get('object', '')), str(c2.get('object', ''))
                    if s1 == s2 and o1 == o2:
                        return ((r1, r2) in opp_pairs or (r2, r1) in opp_pairs)
                    return False
                return matcher
            conditions.append(make_opposite_rel_matcher())
        
        # Pattern C: Subject-specific + different condition tokens (generic pattern)
        if top_subject_count >= max(2, n * 0.3) and not conditions:
            def make_subject_cond_matcher(_subj=top_subject):
                def matcher(c1, c2):
                    s1, s2 = str(c1.get('subject', '')), str(c2.get('subject', ''))
                    if s1 == _subj or s2 == _subj:
                        r1, r2 = str(c1.get('relation', '')), str(c2.get('relation', ''))
                        o1, o2 = str(c1.get('object', '')), str(c2.get('object', ''))
                        if s1 == s2 and r1 == r2 and o1 == o2:
                            c1c = str(c1.get('condition', ''))
                            c2c = str(c2.get('condition', ''))
                            return bool(c1c and c2c and c1c != c2c)
                    return False
                return matcher
            conditions.append(make_subject_cond_matcher())
        
        return conditions if conditions else []
    
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
    
    # ── Test pattern derivation ──
    print("\n" + "="*60)
    print("TEST: Pattern Derivation from Verified Contradictions")
    print("="*60)
    
    sample_verified = pd.DataFrame([
        {
            "claim1_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim1_condition": "GSM8K with GPT-3.5",
            "claim2_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim2_condition": "MATH with Llama-2",
            "contradiction_type": "Experimental_Condition_Contradiction",
            "is_true_positive": True
        },
        {
            "claim1_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim1_condition": "GSM8K with GPT-4",
            "claim2_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim2_condition": "SVAMP with GPT-3.5",
            "contradiction_type": "Experimental_Condition_Contradiction",
            "is_true_positive": True
        },
        {
            "claim1_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim1_condition": "MATH with GPT-3.5",
            "claim2_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim2_condition": "GSM8K with Codex",
            "contradiction_type": "Experimental_Condition_Contradiction",
            "is_true_positive": True
        },
        {
            "claim1_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim1_condition": "GSM8K with Claude",
            "claim2_text": "Chain-of-Thought improves Arithmetic Accuracy",
            "claim2_condition": "GSM8K with PaLM",
            "contradiction_type": "Experimental_Condition_Contradiction",
            "is_true_positive": True
        },
        {
            "claim1_text": "Large Language Model correlates_with Problem Complexity",
            "claim1_condition": "GSM8K with GPT-3.5",
            "claim2_text": "Large Language Model contradicts Problem Complexity",
            "claim2_condition": "MATH with Llama-2",
            "contradiction_type": "Scientific_Dispute",
            "is_true_positive": True
        },
        {
            "claim1_text": "Large Language Model correlates_with Problem Complexity",
            "claim1_condition": "GSM8K with GPT-3.5",
            "claim2_text": "Large Language Model contradicts Problem Complexity",
            "claim2_condition": "GSM8K with Llama-2",
            "contradiction_type": "Scientific_Dispute",
            "is_true_positive": True
        }
    ])
    
    print(f"Sample verified data: {len(sample_verified)} pairs")
    
    # Derive rules
    derived = engine.derive_rules_from_verified(sample_verified, min_support=2)
    print(f"\nDerived {len(derived)} new rules:")
    for rule in derived:
        print(f"  - {rule.name}: {rule.description}")
    
    # Auto add and test
    added = engine.auto_derive_and_add_rules(sample_verified, min_support=2)
    print(f"\nAuto-added {added} rules, total now: {engine.get_rule_summary()['total_rules']}")
    
    # Test derived rule on new pair
    new_claim1 = {
        "subject": "Chain-of-Thought",
        "relation": "improves",
        "object": "Arithmetic Accuracy",
        "condition": "GPQA with GPT-4"
    }
    new_claim2 = {
        "subject": "Chain-of-Thought",
        "relation": "improves",
        "object": "Arithmetic Accuracy",
        "condition": "AQUA with GPT-3.5"
    }
    ctype, conf, rules = engine.analyze_pair(new_claim1, new_claim2)
    print(f"\nDerived rule test on new pair:")
    print(f"  Type: {ctype}")
    print(f"  Confidence: {conf}")
    print(f"  Matched rules: {rules}")
    
    print("\n[PASS] All rule engine tests completed!")
