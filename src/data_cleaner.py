"""
Data cleaning and ontology normalization for extracted claims.
Applies synonym mapping and filters invalid claims.

Author: AI Scientist
Date: 2026-04-20
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from ontology import ONTOLOGY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and normalize extracted claims."""
    
    def __init__(self, ontology=ONTOLOGY):
        """
        Initialize data cleaner with ontology.
        
        Args:
            ontology: Ontology instance for validation
        """
        self.ontology = ontology
        
    @staticmethod
    def _strip_brackets(text: str) -> str:
        """Remove angle brackets and clean whitespace from text."""
        if pd.isna(text):
            return ""
        text = str(text).replace("<", "").replace(">", "").strip()
        return text

    @staticmethod
    def _normalize_condition(cond: str) -> str:
        """Normalize condition field: lowercase, strip, limit length."""
        if pd.isna(cond):
            return ""
        cond = str(cond).strip().lower()
        # Limit to 200 chars
        return cond[:200]

    def normalize_claims(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ontology normalization to claims.
        
        Args:
            claims_df: DataFrame with raw extracted claims
            
        Returns:
            DataFrame with normalized claims
        """
        df = claims_df.copy()

        # Strip brackets from all text fields
        for col in ["subject", "relation", "object", "condition", "evidence_type"]:
            if col in df.columns:
                df[col] = df[col].apply(self._strip_brackets)

        # Normalize condition
        if "condition" in df.columns:
            df["condition"] = df["condition"].apply(self._normalize_condition)

        # Normalize subject
        df["subject_original"] = df["subject"]
        df["subject"] = df["subject"].apply(self.ontology.normalize_concept)
        
        # Normalize relation
        df["relation_original"] = df["relation"]
        df["relation"] = df["relation"].apply(self.ontology.normalize_relation)
        
        logger.info(f"Normalized {len(df)} claims")
        
        return df
    
    def filter_valid_claims(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter claims that match ontology.
        
        Args:
            claims_df: DataFrame with claims
            
        Returns:
            DataFrame with only valid claims
        """
        # Check concept validity
        valid_concepts = claims_df["subject"].apply(self.ontology.is_valid_concept)
        
        # Check relation validity
        valid_relations = claims_df["relation"].apply(self.ontology.is_valid_relation)
        
        # Combined filter
        valid_mask = valid_concepts & valid_relations
        
        valid_df = claims_df[valid_mask].copy()
        
        dropped = len(claims_df) - len(valid_df)
        logger.info(f"Filtered {dropped} invalid claims, {len(valid_df)} remaining")
        
        # Log dropped examples
        if dropped > 0:
            dropped_df = claims_df[~valid_mask]
            logger.info(f"Dropped examples: {dropped_df[['subject', 'relation']].head().to_dict('records')}")
        
        return valid_df
    
    def clean_confidence(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate confidence scores.
        
        Args:
            claims_df: DataFrame with claims
            
        Returns:
            DataFrame with cleaned confidence
        """
        df = claims_df.copy()
        
        def parse_confidence(x):
            try:
                # Extract numeric part
                if isinstance(x, str):
                    # Remove non-numeric characters
                    nums = ''.join(c for c in x if c.isdigit())
                    if nums:
                        return int(nums[0])  # Take first digit
                elif isinstance(x, (int, float)):
                    return int(x)
                return 3  # Default to middle confidence
            except:
                return 3
        
        df["confidence"] = df["confidence"].apply(parse_confidence)
        
        # Clip to valid range
        df["confidence"] = df["confidence"].clip(1, 5)
        
        return df
    
    def clean_evidence_type(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize evidence type values.
        
        Args:
            claims_df: DataFrame with claims
            
        Returns:
            DataFrame with cleaned evidence types
        """
        df = claims_df.copy()
        
        valid_types = ["Experimental_Result", "Theoretical_Analysis", "Literature_Citation"]
        
        def standardize_evidence(x):
            if pd.isna(x):
                return "Experimental_Result"  # Default
            
            x = str(x).strip().lower()
            
            # Map common variations
            if "experiment" in x or "result" in x:
                return "Experimental_Result"
            elif "theory" in x or "analy" in x:
                return "Theoretical_Analysis"
            elif "citation" in x or "literature" in x or "cite" in x:
                return "Literature_Citation"
            else:
                return "Experimental_Result"  # Default
        
        df["evidence_type"] = df["evidence_type"].apply(standardize_evidence)
        
        return df
    
    def remove_duplicates(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate claims from same paper.
        
        Args:
            claims_df: DataFrame with claims
            
        Returns:
            DataFrame with duplicates removed
        """
        # Define duplicate key
        duplicate_cols = ["arxiv_id", "subject", "relation", "object"]
        
        # Keep first occurrence
        df = claims_df.drop_duplicates(subset=duplicate_cols, keep="first")
        
        dropped = len(claims_df) - len(df)
        if dropped > 0:
            logger.info(f"Removed {dropped} duplicate claims")
        
        return df
    
    def process(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full cleaning pipeline.
        
        Args:
            claims_df: Raw extracted claims
            
        Returns:
            Cleaned and normalized claims
        """
        logger.info(f"Starting data cleaning: {len(claims_df)} raw claims")
        
        # Step 1: Normalize
        df = self.normalize_claims(claims_df)
        
        # Step 2: Clean confidence
        df = self.clean_confidence(df)
        
        # Step 3: Clean evidence types
        df = self.clean_evidence_type(df)
        
        # Step 4: Filter valid ontology
        df = self.filter_valid_claims(df)
        
        # Step 5: Remove duplicates
        df = self.remove_duplicates(df)
        
        logger.info(f"Cleaning complete: {len(df)} valid claims")
        
        return df
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """
        Generate statistics about cleaning process.
        
        Args:
            original_df: Original claims
            cleaned_df: Cleaned claims
            
        Returns:
            Dictionary with cleaning statistics
        """
        report = {
            "original_count": len(original_df),
            "cleaned_count": len(cleaned_df),
            "retention_rate": len(cleaned_df) / len(original_df) if len(original_df) > 0 else 0,
            "by_concept": cleaned_df["subject"].value_counts().to_dict(),
            "by_relation": cleaned_df["relation"].value_counts().to_dict(),
            "by_evidence": cleaned_df["evidence_type"].value_counts().to_dict(),
            "avg_confidence": cleaned_df["confidence"].mean()
        }
        
        return report


if __name__ == "__main__":
    # Test with sample data
    sample_claims = pd.DataFrame([
        {
            "arxiv_id": "test1",
            "subject": "GPT-4",
            "relation": "better than",
            "object": "baseline",
            "condition": "GSM8K dataset",
            "evidence_type": "experimental",
            "confidence": "5"
        },
        {
            "arxiv_id": "test1",
            "subject": "CoT",
            "relation": "enhances",
            "object": "accuracy",
            "condition": "math problems",
            "evidence_type": "theory",
            "confidence": "4"
        },
        {
            "arxiv_id": "test2",
            "subject": "InvalidConcept",
            "relation": "unknown_relation",
            "object": "something",
            "condition": "test",
            "evidence_type": "bad_type",
            "confidence": "10"
        }
    ])
    
    cleaner = DataCleaner()
    
    print("Original claims:")
    print(sample_claims)
    
    cleaned = cleaner.process(sample_claims)
    
    print("\nCleaned claims:")
    print(cleaned)
    
    report = cleaner.generate_cleaning_report(sample_claims, cleaned)
    print(f"\nCleaning report: {report}")
