"""
Ontology definition for Chain-of-Thought Mathematical Reasoning domain.
Defines core concepts, relations, and contradiction types.

Author: AI Scientist
Date: 2026-04-20
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import json


@dataclass
class Ontology:
    """Domain ontology for scientific contradiction detection."""
    
    # 10 core concepts (max)
    concepts: List[str] = None
    
    # 10 core relations (max)
    relations: List[str] = None
    
    # 6 contradiction types
    contradiction_types: List[str] = None
    
    # Synonym mappings for normalization
    synonym_map: Dict[str, str] = None
    
    def __post_init__(self):
        if self.concepts is None:
            self.concepts = [
                "Chain-of-Thought",
                "Mathematical Reasoning",
                "Large Language Model",
                "Few-Shot Prompting",
                "Zero-Shot Prompting",
                "Arithmetic Accuracy",
                "Step-by-Step Reasoning",
                "Self-Consistency",
                "Verification Method",
                "Problem Complexity"
            ]
        
        if self.relations is None:
            self.relations = [
                "improves",
                "degrades",
                "correlates_with",
                "outperforms",
                "depends_on",
                "enables",
                "reduces",
                "increases",
                "equals",
                "contradicts"
            ]
        
        if self.contradiction_types is None:
            self.contradiction_types = [
                "Experimental_Condition_Contradiction",
                "Measurement_Method_Contradiction", 
                "Statistical_Error_Contradiction",
                "Scientific_Dispute",
                "Logical_Contradiction",
                "No_Contradiction"
            ]
        
        if self.synonym_map is None:
            self.synonym_map = {
                # Subject/Model synonyms
                "GPT-4": "Large Language Model",
                "GPT-3.5": "Large Language Model",
                "GPT-3": "Large Language Model",
                "Llama 2": "Large Language Model",
                "Llama 3": "Large Language Model",
                "PaLM": "Large Language Model",
                "Claude": "Large Language Model",
                "Codex": "Large Language Model",
                
                # Method synonyms
                "CoT": "Chain-of-Thought",
                "Chain of Thought": "Chain-of-Thought",
                "chain-of-thought": "Chain-of-Thought",
                "COT prompting": "Chain-of-Thought",
                
                # Relation synonyms
                "enhances": "improves",
                "boosts": "improves",
                "better than": "outperforms",
                "superior to": "outperforms",
                "worse than": "degrades",
                "inferior to": "degrades",
                "linked to": "correlates_with",
                "associated with": "correlates_with",
                "requires": "depends_on",
                "needs": "depends_on",
                "facilitates": "enables",
                "allows": "enables",
                "decreases": "reduces",
                "lowers": "reduces",
                "raises": "increases",
                "boosts": "increases",
                "same as": "equals",
                "identical to": "equals",
                "disagrees with": "contradicts",
                "inconsistent with": "contradicts"
            }
    
    def normalize_concept(self, concept: str) -> str:
        """Normalize a concept to standard ontology term."""
        concept_clean = concept.strip()
        return self.synonym_map.get(concept_clean, concept_clean)
    
    def normalize_relation(self, relation: str) -> str:
        """Normalize a relation to standard ontology term."""
        relation_clean = relation.strip().lower()
        return self.synonym_map.get(relation_clean, relation_clean)
    
    def is_valid_concept(self, concept: str) -> bool:
        """Check if concept is in ontology."""
        normalized = self.normalize_concept(concept)
        return normalized in self.concepts
    
    def is_valid_relation(self, relation: str) -> bool:
        """Check if relation is in ontology."""
        normalized = self.normalize_relation(relation)
        return normalized in self.relations
    
    def is_valid_contradiction_type(self, ctype: str) -> bool:
        """Check if contradiction type is valid."""
        return ctype in self.contradiction_types
    
    def to_dict(self) -> Dict:
        """Serialize ontology to dictionary."""
        return {
            "concepts": self.concepts,
            "relations": self.relations,
            "contradiction_types": self.contradiction_types,
            "synonym_map": self.synonym_map
        }
    
    def save(self, filepath: str):
        """Save ontology to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "Ontology":
        """Load ontology from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            concepts=data.get("concepts"),
            relations=data.get("relations"),
            contradiction_types=data.get("contradiction_types"),
            synonym_map=data.get("synonym_map")
        )


# Global ontology instance
ONTOLOGY = Ontology()


if __name__ == "__main__":
    # Test ontology
    print("Testing Ontology...")
    print(f"Concepts: {ONTOLOGY.concepts}")
    print(f"Relations: {ONTOLOGY.relations}")
    print(f"Contradiction Types: {ONTOLOGY.contradiction_types}")
    
    # Test normalization
    print(f"\nNormalization tests:")
    print(f"'GPT-4' -> '{ONTOLOGY.normalize_concept('GPT-4')}'")
    print(f"'CoT' -> '{ONTOLOGY.normalize_concept('CoT')}'")
    print(f"'better than' -> '{ONTOLOGY.normalize_relation('better than')}'")
    
    # Save to file
    ONTOLOGY.save("ontology.json")
    print("\nOntology saved to ontology.json")
