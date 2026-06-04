"""
PDF text extraction and preprocessing using PyMuPDF.
Extracts text from Results, Discussion, Conclusion sections only.

Author: AI Scientist
Date: 2026-04-20
"""

import fitz  # PyMuPDF
import pandas as pd
import os
import re
from typing import Optional, List, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract and clean text from PDF papers."""
    
    # Section keywords to identify core content
    CORE_SECTIONS = [
        "results", "result",
        "discussion", "discussions",
        "conclusion", "conclusions",
        "findings",
        "experimental results",
        "evaluation",
        "experiments"
    ]
    
    # Sections to exclude
    EXCLUDE_SECTIONS = [
        "references",
        "acknowledgments",
        "acknowledgements",
        "appendix",
        "supplementary",
        "funding",
        "conflict of interest"
    ]
    
    def __init__(self,
                 metadata_path: str = "./data/core_papers.csv",
                 pdf_dir: str = "./papers/pdfs",
                 output_path: str = "./data/papers_with_text.csv",
                 min_text_length: int = 100,
                 max_pages: int = 20):
        """
        Initialize text extractor.
        
        Args:
            metadata_path: Path to paper metadata CSV
            pdf_dir: Directory containing PDF files
            output_path: Path to save output CSV
            min_text_length: Minimum text length to consider valid
            max_pages: Maximum pages to process per paper
        """
        self.metadata_path = metadata_path
        self.pdf_dir = pdf_dir
        self.output_path = output_path
        self.min_text_length = min_text_length
        self.max_pages = max_pages
        
        # Load metadata
        self.df = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load paper metadata."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        
        df = pd.read_csv(self.metadata_path)
        logger.info(f"Loaded {len(df)} papers from metadata")
        return df
    
    def _extract_raw_text(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            doc = fitz.open(pdf_path)
            
            # Process limited pages
            pages_to_process = min(len(doc), self.max_pages)
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with single
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with single
        text = re.sub(r' +', ' ', text)
        
        # Remove tabs
        text = text.replace('\t', ' ')
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def _find_core_section_start(self, text: str) -> int:
        """
        Find start index of core sections (Results/Discussion/Conclusion).
        
        Args:
            text: Full text
            
        Returns:
            Start index or 0 if not found
        """
        text_lower = text.lower()
        start_index = 0
        
        # Find earliest occurrence of any core section
        for section in self.CORE_SECTIONS:
            # Match section headers (e.g., "3. Results" or "Results")
            patterns = [
                rf'\n\s*\d+\.?\s*{section}\s*\n',  # Numbered section
                rf'\n\s*{section}\s*\n',  # Plain section
                rf'\n\s*{section.upper()}\s*\n',  # ALL CAPS
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    start_index = max(start_index, match.start())
        
        return start_index
    
    def _find_exclude_section_start(self, text: str) -> int:
        """
        Find start index of sections to exclude (References, etc.).
        
        Args:
            text: Full text
            
        Returns:
            Start index or -1 if not found
        """
        text_lower = text.lower()
        exclude_index = -1
        
        for section in self.EXCLUDE_SECTIONS:
            patterns = [
                rf'\n\s*\d+\.?\s*{section}\s*\n',
                rf'\n\s*{section}\s*\n',
                rf'\n\s*{section.upper()}\s*\n',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if exclude_index == -1:
                        exclude_index = match.start()
                    else:
                        exclude_index = min(exclude_index, match.start())
        
        return exclude_index
    
    def extract_core_text(self, pdf_path: str) -> str:
        """
        Extract text from core sections only.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Core section text
        """
        # Extract raw text
        raw_text = self._extract_raw_text(pdf_path)
        
        if not raw_text:
            return ""
        
        # Clean text
        cleaned = self._clean_text(raw_text)
        
        # Find core section start
        core_start = self._find_core_section_start(cleaned)
        
        # Find exclude section start
        exclude_start = self._find_exclude_section_start(cleaned)
        
        # Extract core text
        if exclude_start > core_start:
            core_text = cleaned[core_start:exclude_start]
        else:
            core_text = cleaned[core_start:]
        
        return core_text.strip()
    
    def process_all_papers(self) -> pd.DataFrame:
        """
        Process all papers and extract text.
        
        Returns:
            DataFrame with extracted text
        """
        results = []
        
        logger.info(f"Processing {len(self.df)} papers...")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting text"):
            arxiv_id = row["arxiv_id"]
            pdf_path = os.path.join(self.pdf_dir, f"{arxiv_id}.pdf")
            
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not found: {pdf_path}")
                results.append({
                    "arxiv_id": arxiv_id,
                    "full_text": "",
                    "text_extracted": False,
                    "text_length": 0
                })
                continue
            
            # Extract text
            text = self.extract_core_text(pdf_path)
            
            results.append({
                "arxiv_id": arxiv_id,
                "full_text": text,
                "text_extracted": len(text) >= self.min_text_length,
                "text_length": len(text)
            })
        
        # Merge with metadata
        results_df = pd.DataFrame(results)
        self.df = self.df.merge(results_df, on="arxiv_id", how="left")
        
        # Filter valid papers
        valid_papers = self.df[self.df["text_extracted"]].copy()
        
        logger.info(f"Successfully extracted text from {len(valid_papers)}/{len(self.df)} papers")
        logger.info(f"Average text length: {valid_papers['text_length'].mean():.0f} chars")
        
        return self.df
    
    def save(self, df: Optional[pd.DataFrame] = None):
        """
        Save results to CSV.
        
        Args:
            df: DataFrame to save (uses self.df if None)
        """
        if df is None:
            df = self.df
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        logger.info(f"Saved to {self.output_path}")
    
    def run(self) -> pd.DataFrame:
        """
        Execute full extraction pipeline.
        
        Returns:
            DataFrame with extracted text
        """
        df = self.process_all_papers()
        self.save(df)
        return df


def load_papers_with_text(path: str = "./data/papers_with_text.csv") -> pd.DataFrame:
    """Load papers with extracted text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize extractor
    extractor = TextExtractor(
        metadata_path=os.path.join(config["data"]["base_path"], "core_papers.csv"),
        pdf_dir=config["data"]["pdf_path"],
        output_path=os.path.join(config["data"]["base_path"], "papers_with_text.csv"),
        min_text_length=config.get("extraction", {}).get("min_text_length", 100)
    )
    
    # Run
    df = extractor.run()
    print(f"\nExtraction complete: {df['text_extracted'].sum()}/{len(df)} papers")
