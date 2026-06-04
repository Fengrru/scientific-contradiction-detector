"""
Main orchestration script for Scientific Contradiction Detection System.
Executes full pipeline from data collection to paper generation.

Author: AI Scientist
Date: 2026-04-20
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure log directory exists before configuring logging
Path("./logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

from paper_fetcher import PaperFetcher
from pdf_downloader import PDFDownloader
from text_extractor import TextExtractor
from batch_processor import BatchProcessor
from contradiction_detector import ContradictionDetector, manual_verification_sample
from rule_engine import RuleEngine, replace_llm_with_rules
from paper_generator import PaperGenerator, generate_paper_summary
from llm_client import LLMClient


class Pipeline:
    """End-to-end contradiction detection pipeline."""
    
    def __init__(self, config_path: str = "./config.yaml", api_key: str = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML
            api_key: DeepSeek API key (overrides config)
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize LLM client
        ds_api_key = api_key or self.config.get("api", {}).get("deepseek_api_key")
        self.llm_client = LLMClient(
            api_key=ds_api_key,
            model=self.config.get("api", {}).get("model", "deepseek-chat"),
            temperature=self.config["model"]["temperature"],
            max_tokens=2048
        ) if ds_api_key else None
        
        # Ensure directories exist
        Path("./data").mkdir(exist_ok=True)
        Path("./papers/pdfs").mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)
        Path("./results").mkdir(exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML, resolving ${ENV_VAR} patterns."""
        import re
        def _resolve(obj):
            if isinstance(obj, str):
                return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
            if isinstance(obj, dict):
                return {k: _resolve(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_resolve(v) for v in obj]
            return obj
        with open(self.config_path, "r") as f:
            return _resolve(yaml.safe_load(f))
    
    def run_phase_1(self, skip_download: bool = False) -> bool:
        """
        Execute Phase 1: Data Collection and Text Extraction.
        
        Args:
            skip_download: Skip PDF download if already done
            
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Collection and Text Extraction")
        logger.info("=" * 60)
        
        try:
            # Step 1: Fetch paper metadata
            logger.info("Step 1/3: Fetching paper metadata...")
            fetcher = PaperFetcher(
                search_query=self.config["data"]["search_query"],
                start_year=self.config["data"]["start_year"],
                max_papers=self.config["data"]["max_papers"],
                semantic_scholar_api_key=self.config["data"].get("semantic_scholar_api_key"),
                base_path=self.config["data"]["base_path"]
            )
            papers_df = fetcher.run()
            logger.info(f"Retrieved {len(papers_df)} papers with metadata")
            
            # Step 2: Download PDFs
            if not skip_download:
                logger.info("Step 2/3: Downloading PDFs...")
                downloader = PDFDownloader(
                    metadata_path=os.path.join(self.config["data"]["base_path"], "core_papers.csv"),
                    output_dir=self.config["data"]["pdf_path"],
                    delay=self.config["rate_limits"]["arxiv_download_delay"]
                )
                stats = downloader.download_all()
                logger.info(f"Download complete: {stats}")
            else:
                logger.info("Step 2/3: Skipping PDF download (already done)")
            
            # Step 3: Extract text
            logger.info("Step 3/3: Extracting text from PDFs...")
            extractor = TextExtractor(
                metadata_path=os.path.join(self.config["data"]["base_path"], "core_papers.csv"),
                pdf_dir=self.config["data"]["pdf_path"],
                output_path=os.path.join(self.config["data"]["base_path"], "papers_with_text.csv"),
                min_text_length=100
            )
            papers_with_text = extractor.run()
            
            success_count = papers_with_text["text_extracted"].sum()
            logger.info(f"Text extraction complete: {success_count}/{len(papers_with_text)} papers")
            
            if success_count < 150:
                logger.warning(f"Only {success_count} papers with text (target: 150+)")
            
            logger.info("Phase 1 complete!")
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}", exc_info=True)
            return False
    
    def run_phase_2(self, max_papers: int = None, mock_mode: bool = False) -> bool:
        """
        Execute Phase 2: Claim Extraction and Cleaning.
        
        Args:
            max_papers: Limit papers for testing
            mock_mode: Use mock extraction without loading LLM
            
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Claim Extraction and Cleaning")
        if mock_mode:
            logger.info("(MOCK MODE - No LLM loaded)")
        logger.info("=" * 60)
        
        try:
            logger.info("Initializing batch processor...")
            processor = BatchProcessor(
                papers_path=os.path.join(self.config["data"]["base_path"], "papers_with_text.csv"),
                output_dir=self.config["data"]["base_path"],
                model_name=self.config["model"]["name"],
                use_4bit=self.config["model"]["quantization"] == "4bit",
                batch_size=self.config["output"]["backup_frequency"],
                mock_mode=mock_mode,
                api_client=self.llm_client
            )
            
            logger.info("Running extraction pipeline...")
            cleaned_claims = processor.run(max_papers=max_papers)
            
            if len(cleaned_claims) < 3000:
                logger.warning(f"Only {len(cleaned_claims)} claims extracted (target: 3000+)")
            else:
                logger.info(f"Successfully extracted {len(cleaned_claims)} claims")
            
            logger.info("Phase 2 complete!")
            return True
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}", exc_info=True)
            return False
    
    def run_phase_3(self, mock_mode: bool = False) -> bool:
        """
        Execute Phase 3: Contradiction Detection.
        
        Args:
            mock_mode: Use mock mode (no LLM required)
            
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Contradiction Detection")
        if mock_mode:
            logger.info("(MOCK MODE - No LLM loaded)")
        logger.info("=" * 60)
        
        try:
            # Load cleaned claims
            import pandas as pd
            claims_path = os.path.join(self.config["data"]["base_path"], "cleaned_claims.csv")
            
            if not os.path.exists(claims_path):
                logger.error(f"Cleaned claims not found: {claims_path}")
                return False
            
            claims_df = pd.read_csv(claims_path)
            logger.info(f"Loaded {len(claims_df)} cleaned claims")
            
            # Initialize detector
            detector = ContradictionDetector(
                model_name=self.config["model"]["name"],
                use_4bit=self.config["model"]["quantization"] == "4bit",
                citation_weight=self.config["scoring"]["citation_weight"],
                confidence_weight=self.config["scoring"]["confidence_weight"],
                mock_mode=mock_mode,
                api_client=self.llm_client
            )
            
            # Run detection
            logger.info("Running contradiction detection...")
            contradictions = detector.run(claims_df)
            
            # Save results
            output_path = os.path.join(self.config["data"]["base_path"], "final_contradictions.csv")
            contradictions.to_csv(output_path, index=False)
            logger.info(f"Saved {len(contradictions)} contradictions to {output_path}")
            
            # Prepare manual verification sample
            verification_df = manual_verification_sample(
                contradictions,
                n_samples=min(100, len(contradictions)),
                output_path=os.path.join(self.config["data"]["base_path"], "manual_verification.csv")
            )
            
            # Initialize rule engine
            logger.info("Initializing rule engine...")
            engine = RuleEngine()
            
            # Apply rules to candidates
            candidates_path = os.path.join(self.config["data"]["base_path"], "candidate_contradictions.csv")
            if os.path.exists(candidates_path):
                candidates = pd.read_csv(candidates_path)
                hybrid_results = replace_llm_with_rules(candidates, engine)
                hybrid_results.to_csv(candidates_path, index=False)
                logger.info(f"Applied rule engine to {len(hybrid_results)} candidates")
            
            logger.info("Phase 3 complete!")
            return True
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}", exc_info=True)
            return False
    
    def run_phase_4(self) -> bool:
        """
        Execute Phase 4: Paper Generation and Release Preparation.
        
        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: Paper Generation and Release")
        logger.info("=" * 60)
        
        try:
            import pandas as pd
            
            # Load data
            claims_path = os.path.join(self.config["data"]["base_path"], "cleaned_claims.csv")
            contradictions_path = os.path.join(self.config["data"]["base_path"], "final_contradictions.csv")
            papers_path = os.path.join(self.config["data"]["base_path"], "core_papers.csv")
            
            claims_df = pd.read_csv(claims_path) if os.path.exists(claims_path) else pd.DataFrame()
            contradictions_df = pd.read_csv(contradictions_path) if os.path.exists(contradictions_path) else pd.DataFrame()
            papers_df = pd.read_csv(papers_path) if os.path.exists(papers_path) else pd.DataFrame()
            
            # Generate paper
            logger.info("Generating research paper...")
            generator = PaperGenerator(
                domain=self.config["data"]["search_query"],
                target_venue=self.config["paper"]["target_conference"]
            )
            
            paper_text = generator.generate_full_paper(contradictions_df, claims_df, papers_df)
            
            paper_path = "./papers/contradiction_paper.tex"
            generator.save_paper(paper_text, paper_path)
            logger.info(f"Paper saved to {paper_path}")
            
            # Generate README summary
            summary = generate_paper_summary(contradictions_df)
            readme_path = "./results/README_SUMMARY.md"
            with open(readme_path, "w") as f:
                f.write(summary)
            logger.info(f"Summary saved to {readme_path}")
            
            logger.info("Phase 4 complete!")
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}", exc_info=True)
            return False
    
    def run_all(self, skip_download: bool = False, max_papers: int = None) -> bool:
        """
        Execute complete pipeline.
        
        Args:
            skip_download: Skip PDF download
            max_papers: Limit papers for testing
            
        Returns:
            True if all phases successful
        """
        logger.info("Starting full pipeline execution...")
        
        results = []
        
        # Phase 1
        results.append(("Phase 1", self.run_phase_1(skip_download=skip_download)))
        
        # Phase 2
        results.append(("Phase 2", self.run_phase_2(max_papers=max_papers, mock_mode=False)))
        
        # Phase 3
        results.append(("Phase 3", self.run_phase_3()))
        
        # Phase 4
        results.append(("Phase 4", self.run_phase_4()))
        
        # Summary
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        for phase, success in results:
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"{phase}: {status}")
        
        all_success = all(s for _, s in results)
        
        if all_success:
            logger.info("All phases completed successfully!")
        else:
            logger.warning("Some phases failed. Check logs for details.")
        
        return all_success


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scientific Contradiction Detection Pipeline"
    )
    
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Run specific phase or all"
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download (assume already done)"
    )
    
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers for testing"
    )
    
    parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode for Phase 2 (no LLM required)"
    )
    
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key for LLM calls"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = Pipeline(config_path=args.config, api_key=args.api_key)
    
    # Execute requested phase
    if args.phase == "1":
        success = pipeline.run_phase_1(skip_download=args.skip_download)
    elif args.phase == "2":
        success = pipeline.run_phase_2(max_papers=args.max_papers, mock_mode=args.mock)
    elif args.phase == "3":
        success = pipeline.run_phase_3(mock_mode=args.mock)
    elif args.phase == "4":
        success = pipeline.run_phase_4()
    else:
        success = pipeline.run_all(
            skip_download=args.skip_download,
            max_papers=args.max_papers
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
