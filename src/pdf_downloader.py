"""
PDF batch downloader with rate limiting.
Downloads paper PDFs from arXiv with configurable delay.

Author: AI Scientist
Date: 2026-04-20
"""

import requests
import pandas as pd
import os
import time
from typing import Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDownloader:
    """Download PDFs from arXiv with rate limiting."""
    
    def __init__(self,
                 metadata_path: str = "./data/core_papers.csv",
                 output_dir: str = "./papers/pdfs",
                 delay: int = 15,
                 timeout: int = 30):
        """
        Initialize PDF downloader.
        
        Args:
            metadata_path: Path to CSV with paper metadata
            output_dir: Directory to save PDFs
            delay: Seconds between downloads (rate limiting)
            timeout: HTTP request timeout
        """
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.delay = delay
        self.timeout = timeout
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metadata
        self.df = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load paper metadata from CSV."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        df = pd.read_csv(self.metadata_path)
        logger.info(f"Loaded {len(df)} papers from metadata")
        return df
    
    def _download_pdf(self, arxiv_id: str, pdf_url: str) -> bool:
        """
        Download single PDF.
        
        Args:
            arxiv_id: arXiv paper ID
            pdf_url: URL to PDF
            
        Returns:
            True if successful
        """
        output_path = os.path.join(self.output_dir, f"{arxiv_id}.pdf")
        
        # Skip if already exists
        if os.path.exists(output_path):
            logger.debug(f"Skipping {arxiv_id}, already downloaded")
            return True
        
        try:
            response = requests.get(pdf_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {arxiv_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {arxiv_id}: {e}")
            return False
    
    def download_all(self, resume: bool = True) -> dict:
        """
        Download all PDFs with rate limiting.
        
        Args:
            resume: Skip already downloaded files
            
        Returns:
            Statistics dict with success/failure counts
        """
        stats = {"success": 0, "failed": 0, "skipped": 0, "total": len(self.df)}
        
        logger.info(f"Starting download of {stats['total']} papers")
        logger.info(f"Rate limit: {self.delay}s between downloads")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Downloading PDFs"):
            arxiv_id = row["arxiv_id"]
            pdf_url = row["pdf_url"]
            output_path = os.path.join(self.output_dir, f"{arxiv_id}.pdf")
            
            # Check if already exists
            if resume and os.path.exists(output_path):
                stats["skipped"] += 1
                continue
            
            # Download
            success = self._download_pdf(arxiv_id, pdf_url)
            
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            
            # Rate limiting (except for last item)
            if idx < len(self.df) - 1:
                time.sleep(self.delay)
        
        logger.info(f"Download complete: {stats}")
        return stats
    
    def verify_downloads(self) -> pd.DataFrame:
        """
        Verify which papers have been downloaded.
        
        Returns:
            DataFrame with download status column
        """
        self.df["pdf_downloaded"] = self.df["arxiv_id"].apply(
            lambda x: os.path.exists(os.path.join(self.output_dir, f"{x}.pdf"))
        )
        
        downloaded = self.df["pdf_downloaded"].sum()
        logger.info(f"Verified: {downloaded}/{len(self.df)} papers downloaded")
        
        return self.df
    
    def get_missing_pdfs(self) -> pd.DataFrame:
        """Get DataFrame of papers without downloaded PDFs."""
        self.verify_downloads()
        missing = self.df[~self.df["pdf_downloaded"]].copy()
        
        if len(missing) > 0:
            logger.warning(f"{len(missing)} papers missing PDFs")
        
        return missing


def download_missing_papers(metadata_path: str = "./data/core_papers.csv",
                             pdf_dir: str = "./papers/pdfs",
                             delay: int = 15) -> dict:
    """
    Convenience function to download only missing PDFs.
    
    Args:
        metadata_path: Path to metadata CSV
        pdf_dir: Directory for PDFs
        delay: Download delay
        
    Returns:
        Download statistics
    """
    downloader = PDFDownloader(
        metadata_path=metadata_path,
        output_dir=pdf_dir,
        delay=delay
    )
    
    return downloader.download_all(resume=True)


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize and run
    downloader = PDFDownloader(
        metadata_path=os.path.join(config["data"]["base_path"], "core_papers.csv"),
        output_dir=config["data"]["pdf_path"],
        delay=config["rate_limits"]["arxiv_download_delay"]
    )
    
    stats = downloader.download_all()
    print(f"\nDownload statistics: {stats}")
