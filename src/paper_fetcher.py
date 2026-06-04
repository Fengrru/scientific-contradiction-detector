"""
Paper metadata fetcher integrating arXiv and Semantic Scholar APIs.
Retrieves paper metadata and citation counts for contradiction detection.

Author: AI Scientist
Date: 2026-04-20
"""

import arxiv
import semanticscholar as sch
import pandas as pd
import time
import os
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperFetcher:
    """Fetch paper metadata from arXiv and Semantic Scholar."""
    
    def __init__(self, 
                 search_query: str,
                 start_year: int = 2021,
                 max_papers: int = 200,
                 semantic_scholar_api_key: Optional[str] = None,
                 base_path: str = "./data"):
        """
        Initialize paper fetcher.
        
        Args:
            search_query: Search query for papers
            start_year: Only include papers from this year onwards
            max_papers: Maximum papers to retrieve
            semantic_scholar_api_key: API key for Semantic Scholar
            base_path: Directory to save data
        """
        self.search_query = search_query
        self.start_year = start_year
        self.max_papers = max_papers
        self.base_path = base_path
        self.semantic_scholar_api_key = semantic_scholar_api_key
        
        # Create output directory
        os.makedirs(base_path, exist_ok=True)
        
        # Initialize clients
        self.arxiv_client = arxiv.Client()
        self.sch_client = sch.SemanticScholar(api_key=semantic_scholar_api_key) if semantic_scholar_api_key else None
        
    def fetch_arxiv_metadata(self) -> List[Dict]:
        """
        Fetch paper metadata from arXiv.
        
        Returns:
            List of paper dictionaries with metadata
        """
        logger.info(f"Fetching papers from arXiv with query: '{self.search_query}'")
        
        search = arxiv.Search(
            query=self.search_query,
            max_results=self.max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in tqdm(self.arxiv_client.results(search), total=self.max_papers, desc="Fetching arXiv"):
            if result.published.year >= self.start_year:
                paper = {
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.date().isoformat(),
                    "citations": 0,
                    "pdf_url": result.pdf_url,
                    "primary_category": result.primary_category
                }
                papers.append(paper)
        
        logger.info(f"Retrieved {len(papers)} papers from arXiv")
        return papers
    
    def supplement_citations(self, papers: List[Dict]) -> List[Dict]:
        """
        Supplement papers with citation counts from Semantic Scholar.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Papers with citation counts added
        """
        if not self.sch_client:
            logger.warning("No Semantic Scholar API key provided, skipping citation supplement")
            return papers
        
        logger.info("Supplementing citation counts from Semantic Scholar...")
        
        for i, paper in enumerate(tqdm(papers, desc="Fetching citations")):
            try:
                # Rate limiting
                time.sleep(1)
                
                # Query Semantic Scholar
                sch_paper = self.sch_client.get_paper(f"ARXIV:{paper['arxiv_id']}")
                papers[i]["citations"] = sch_paper.citationCount if hasattr(sch_paper, 'citationCount') else 0
                
            except Exception as e:
                logger.warning(f"Failed to get citations for {paper['arxiv_id']}: {e}")
                papers[i]["citations"] = 0
                continue
        
        return papers
    
    def filter_and_sort(self, papers: List[Dict]) -> pd.DataFrame:
        """
        Filter papers by quality and sort by citation count.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            DataFrame of top papers sorted by citations
        """
        df = pd.DataFrame(papers)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["arxiv_id"], keep="first")
        
        # Sort by citations descending
        df = df.sort_values("citations", ascending=False)
        
        # Keep top max_papers
        df = df.head(self.max_papers).reset_index(drop=True)
        
        logger.info(f"Final dataset: {len(df)} papers")
        logger.info(f"Citation range: {df['citations'].min()} - {df['citations'].max()}")
        
        return df
    
    def save_metadata(self, df: pd.DataFrame, filename: str = "core_papers.csv"):
        """
        Save paper metadata to CSV.
        
        Args:
            df: DataFrame with paper metadata
            filename: Output filename
        """
        filepath = os.path.join(self.base_path, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Saved metadata to {filepath}")
    
    def run(self) -> pd.DataFrame:
        """
        Execute full fetch pipeline.
        
        Returns:
            DataFrame with paper metadata
        """
        # Step 1: Fetch from arXiv
        papers = self.fetch_arxiv_metadata()
        
        # Step 2: Supplement citations
        papers = self.supplement_citations(papers)
        
        # Step 3: Filter and sort
        df = self.filter_and_sort(papers)
        
        # Step 4: Save
        self.save_metadata(df)
        
        return df


def load_papers_metadata(filepath: str = "./data/core_papers.csv") -> pd.DataFrame:
    """Load previously saved paper metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    return pd.read_csv(filepath)


if __name__ == "__main__":
    # Test the fetcher
    import yaml
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize fetcher
    fetcher = PaperFetcher(
        search_query=config["data"]["search_query"],
        start_year=config["data"]["start_year"],
        max_papers=config["data"]["max_papers"],
        semantic_scholar_api_key=config["data"].get("semantic_scholar_api_key"),
        base_path=config["data"]["base_path"]
    )
    
    # Run
    df = fetcher.run()
    print(f"\nDataset preview:")
    print(df[["arxiv_id", "title", "citations"]].head())
