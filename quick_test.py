"""
Quick test pipeline with 10 papers only.
Tests full workflow: download -> extract text -> extract claims.

Author: AI Scientist
Date: 2026-04-20
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from paper_fetcher import PaperFetcher
from pdf_downloader import PDFDownloader
from text_extractor import TextExtractor
from ontology import ONTOLOGY

def main():
    print("="*60)
    print("QUICK TEST PIPELINE (10 papers only)")
    print("="*60)
    
    # Step 1: Load existing metadata and limit to 10 papers
    print("\n[Step 1] Loading metadata...")
    df = pd.read_csv("./data/core_papers.csv")
    df_10 = df.head(10).copy()
    df_10.to_csv("./data/core_papers_10.csv", index=False)
    print(f"  Selected {len(df_10)} papers for quick test")
    
    # Step 2: Download 10 PDFs
    print("\n[Step 2] Downloading 10 PDFs...")
    downloader = PDFDownloader(
        metadata_path="./data/core_papers_10.csv",
        output_dir="./papers/pdfs",
        delay=5,  # Faster for testing
        timeout=30
    )
    stats = downloader.download_all(resume=False)
    print(f"  Download stats: {stats}")
    
    # Step 3: Extract text
    print("\n[Step 3] Extracting text from PDFs...")
    extractor = TextExtractor(
        metadata_path="./data/core_papers_10.csv",
        pdf_dir="./papers/pdfs",
        output_path="./data/papers_with_text_10.csv",
        min_text_length=100
    )
    result_df = extractor.run()
    success_count = result_df["text_extracted"].sum()
    print(f"  Text extracted from {success_count}/{len(result_df)} papers")
    
    # Step 4: Show sample
    print("\n[Step 4] Sample extracted text:")
    for idx, row in result_df.iterrows():
        if row["text_extracted"]:
            text_preview = row["full_text"][:200] if len(str(row["full_text"])) > 200 else row["full_text"]
            print(f"\n  Paper {row['arxiv_id']}:")
            print(f"  {text_preview}...")
            break
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - ./data/core_papers_10.csv (10 paper metadata)")
    print(f"  - ./data/papers_with_text_10.csv (extracted text)")
    print(f"  - ./papers/pdfs/ (downloaded PDFs)")

if __name__ == "__main__":
    main()
