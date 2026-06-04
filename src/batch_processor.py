"""
Batch processing pipeline for claim extraction from all papers.
Integrates paper loading, LLM extraction, and result saving.

Author: AI Scientist
Date: 2026-04-20
"""

import pandas as pd
import os
import sys
from typing import Optional
import logging
from tqdm import tqdm

from claim_extractor import ClaimExtractor
from data_cleaner import DataCleaner
from ontology import ONTOLOGY
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process all papers to extract and clean claims."""
    
    def __init__(self,
                 papers_path: str = "./data/papers_with_text.csv",
                 output_dir: str = "./data",
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 use_4bit: bool = True,
                 batch_size: int = 10,
                 mock_mode: bool = False,
                 api_client: Optional[LLMClient] = None):
        """
        Initialize batch processor.
        
        Args:
            papers_path: Path to papers with extracted text
            output_dir: Directory for output files
            model_name: LLM model name
            use_4bit: Use 4-bit quantization
            batch_size: Save intermediate results every N papers
            mock_mode: If True, use mock claim extraction without LLM
            api_client: LLMClient for API-based extraction
        """
        self.papers_path = papers_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing claim extractor...")
        self.extractor = ClaimExtractor(
            model_name=model_name,
            use_4bit=use_4bit,
            mock_mode=mock_mode,
            api_client=api_client
        )
        
        logger.info("Initializing data cleaner...")
        self.cleaner = DataCleaner(ontology=ONTOLOGY)
        
        # Load papers
        self.papers_df = self._load_papers()
        
    def _load_papers(self) -> pd.DataFrame:
        """Load papers with extracted text."""
        if not os.path.exists(self.papers_path):
            raise FileNotFoundError(f"Papers file not found: {self.papers_path}")
        
        df = pd.read_csv(self.papers_path)
        
        # Filter papers with text
        df = df[df["text_extracted"] == True].copy()
        
        logger.info(f"Loaded {len(df)} papers with extracted text")
        return df
    
    def process_papers(self, max_papers: Optional[int] = None) -> pd.DataFrame:
        """
        Process all papers to extract claims.
        
        Args:
            max_papers: Limit to N papers (for testing)
            
        Returns:
            DataFrame with all extracted claims (raw)
        """
        papers_to_process = self.papers_df.head(max_papers) if max_papers else self.papers_df
        
        logger.info(f"Processing {len(papers_to_process)} papers...")
        
        # Extract claims
        raw_claims_df = self.extractor.batch_extract(
            papers_df=papers_to_process,
            concepts=ONTOLOGY.concepts,
            relations=ONTOLOGY.relations,
            text_column="full_text",
            save_every=self.batch_size
        )
        
        # Save raw claims
        raw_path = os.path.join(self.output_dir, "all_claims_raw.csv")
        raw_claims_df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw claims to {raw_path}")
        
        return raw_claims_df
    
    def clean_claims(self, raw_claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize extracted claims.
        
        Args:
            raw_claims_df: Raw extracted claims
            
        Returns:
            Cleaned claims DataFrame
        """
        logger.info("Cleaning claims...")
        
        cleaned_df = self.cleaner.process(raw_claims_df)
        
        # Save cleaned claims
        cleaned_path = os.path.join(self.output_dir, "cleaned_claims.csv")
        cleaned_df.to_csv(cleaned_path, index=False)
        logger.info(f"Saved cleaned claims to {cleaned_path}")
        
        # Generate report
        report = self.cleaner.generate_cleaning_report(raw_claims_df, cleaned_df)
        
        logger.info("Cleaning report:")
        for key, value in report.items():
            logger.info(f"  {key}: {value}")
        
        return cleaned_df
    
    def run(self, max_papers: Optional[int] = None) -> pd.DataFrame:
        """
        Execute full pipeline.
        
        Args:
            max_papers: Limit for testing
            
        Returns:
            Final cleaned claims DataFrame
        """
        # Step 1: Extract claims
        raw_claims = self.process_papers(max_papers=max_papers)
        
        # Step 2: Clean claims
        cleaned_claims = self.clean_claims(raw_claims)
        
        logger.info(f"Pipeline complete: {len(cleaned_claims)} valid claims from {len(self.papers_df)} papers")
        
        return cleaned_claims


def quick_test():
    """Quick test with small sample."""
    processor = BatchProcessor()
    
    # Process only 3 papers for testing
    result = processor.run(max_papers=3)
    
    print(f"\nTest complete. Extracted {len(result)} claims.")
    print(result.head())
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process papers for claim extraction")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of papers to process")
    parser.add_argument("--test", action="store_true", help="Run quick test with 3 papers")
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    else:
        processor = BatchProcessor()
        result = processor.run(max_papers=args.max_papers)
        print(f"\nProcessing complete: {len(result)} claims extracted")
