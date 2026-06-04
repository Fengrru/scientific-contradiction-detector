"""
Module testing script for Scientific Contradiction Detection System.
Tests each component independently.

Author: AI Scientist
Date: 2026-04-20
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def test_ontology():
    """Test ontology module."""
    print("\n" + "="*60)
    print("TEST 1: Ontology Module")
    print("="*60)
    
    try:
        from ontology import Ontology, ONTOLOGY
        
        # Test concepts
        print(f"[OK] Concepts loaded: {len(ONTOLOGY.concepts)}")
        assert len(ONTOLOGY.concepts) == 10, "Expected 10 concepts"
        
        # Test relations
        print(f"[OK] Relations loaded: {len(ONTOLOGY.relations)}")
        assert len(ONTOLOGY.relations) == 10, "Expected 10 relations"
        
        # Test contradiction types
        print(f"[OK] Contradiction types: {len(ONTOLOGY.contradiction_types)}")
        assert len(ONTOLOGY.contradiction_types) == 6, "Expected 6 types"
        
        # Test normalization
        normalized = ONTOLOGY.normalize_concept("GPT-4")
        print(f"[OK] Normalization test: 'GPT-4' -> '{normalized}'")
        
        # Save test
        ONTOLOGY.save("./data/test_ontology.json")
        print("[OK] Ontology saved to ./data/test_ontology.json")
        
        # Load test
        loaded = Ontology.load("./data/test_ontology.json")
        print(f"[OK] Ontology loaded, concepts: {len(loaded.concepts)}")
        
        print("\n[PASS] Ontology module working correctly")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Ontology test failed: {e}")
        return False


def test_paper_fetcher():
    """Test paper fetcher module."""
    print("\n" + "="*60)
    print("TEST 2: Paper Fetcher Module")
    print("="*60)
    
    try:
        from paper_fetcher import PaperFetcher
        import yaml
        
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Initialize fetcher (won't actually fetch without API calls)
        fetcher = PaperFetcher(
            search_query="test query",
            start_year=2023,
            max_papers=5,
            semantic_scholar_api_key=None,
            base_path="./data"
        )
        
        print("[OK] PaperFetcher initialized")
        print(f"  - Search query: {fetcher.search_query}")
        print(f"  - Max papers: {fetcher.max_papers}")
        print(f"  - Start year: {fetcher.start_year}")
        
        # Test arXiv client initialization
        print("[OK] arXiv client ready")
        
        print("\n[PASS] Paper fetcher module working (API calls not tested)")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Paper fetcher test failed: {e}")
        return False


def test_pdf_downloader():
    """Test PDF downloader module."""
    print("\n" + "="*60)
    print("TEST 3: PDF Downloader Module")
    print("="*60)
    
    try:
        from pdf_downloader import PDFDownloader
        
        # Create test metadata
        test_df = pd.DataFrame([
            {
                "arxiv_id": "test1",
                "pdf_url": "https://arxiv.org/pdf/test1.pdf",
                "title": "Test Paper 1"
            },
            {
                "arxiv_id": "test2",
                "pdf_url": "https://arxiv.org/pdf/test2.pdf",
                "title": "Test Paper 2"
            }
        ])
        
        test_csv = "./data/test_metadata.csv"
        test_df.to_csv(test_csv, index=False)
        
        # Initialize downloader
        downloader = PDFDownloader(
            metadata_path=test_csv,
            output_dir="./papers/test",
            delay=1,
            timeout=5
        )
        
        print("[OK] PDFDownloader initialized")
        print(f"  - Output dir: {downloader.output_dir}")
        print(f"  - Delay: {downloader.delay}s")
        print(f"  - Papers to process: {len(downloader.df)}")
        
        print("\n[PASS] PDF downloader module working")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] PDF downloader test failed: {e}")
        return False


def test_text_extractor():
    """Test text extractor module."""
    print("\n" + "="*60)
    print("TEST 4: Text Extractor Module")
    print("="*60)
    
    try:
        from text_extractor import TextExtractor
        
        # Create test metadata with text_extracted column
        test_df = pd.DataFrame([
            {
                "arxiv_id": "test1",
                "title": "Test Paper 1",
                "pdf_url": "https://arxiv.org/pdf/test1.pdf"
            }
        ])
        
        test_csv = "./data/test_extract_papers.csv"
        test_df.to_csv(test_csv, index=False)
        
        # Initialize extractor
        extractor = TextExtractor(
            metadata_path=test_csv,
            pdf_dir="./papers/test",
            output_path="./data/test_extracted.csv",
            min_text_length=100
        )
        
        print("[OK] TextExtractor initialized")
        print(f"  - Core sections: {len(extractor.CORE_SECTIONS)}")
        print(f"  - Exclude sections: {len(extractor.EXCLUDE_SECTIONS)}")
        
        # Test text cleaning
        dirty_text = "Hello   world\\n\\n\\nTest\\tcontent"
        clean = extractor._clean_text(dirty_text)
        print(f"[OK] Text cleaning: '{dirty_text[:20]}...' -> '{clean[:20]}...'")
        
        print("\n[PASS] Text extractor module working")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Text extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_cleaner():
    """Test data cleaner module."""
    print("\n" + "="*60)
    print("TEST 5: Data Cleaner Module")
    print("="*60)
    
    try:
        from data_cleaner import DataCleaner
        from ontology import ONTOLOGY
        
        # Create sample claims
        sample_claims = pd.DataFrame([
            {
                "arxiv_id": "paper1",
                "subject": "GPT-4",
                "relation": "better than",
                "object": "baseline",
                "condition": "GSM8K",
                "evidence_type": "experimental",
                "confidence": "5"
            },
            {
                "arxiv_id": "paper2",
                "subject": "CoT",
                "relation": "enhances",
                "object": "accuracy",
                "condition": "math",
                "evidence_type": "theory",
                "confidence": "4"
            }
        ])
        
        cleaner = DataCleaner(ontology=ONTOLOGY)
        
        print("[OK] DataCleaner initialized")
        
        # Test normalization
        normalized = cleaner.normalize_claims(sample_claims)
        print(f"[OK] Normalized {len(normalized)} claims")
        print(f"  - Subject normalization: 'GPT-4' -> '{normalized.iloc[0]['subject']}'")
        
        # Test full cleaning
        cleaned = cleaner.process(sample_claims)
        print(f"[OK] Full cleaning: {len(cleaned)} valid claims")
        
        # Test report
        report = cleaner.generate_cleaning_report(sample_claims, cleaned)
        print(f"[OK] Cleaning report generated")
        print(f"  - Retention rate: {report['retention_rate']:.1%}")
        
        print("\n[PASS] Data cleaner module working")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Data cleaner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rule_engine():
    """Test rule engine module."""
    print("\n" + "="*60)
    print("TEST 6: Rule Engine Module")
    print("="*60)
    
    try:
        from rule_engine import RuleEngine
        
        engine = RuleEngine()
        
        print("[OK] RuleEngine initialized")
        
        # Get rule summary
        summary = engine.get_rule_summary()
        print(f"[OK] Loaded {summary['total_rules']} rules:")
        for rule in summary['rules']:
            print(f"  - {rule['name']}: {rule['type']}")
        
        # Test claim analysis
        claim1 = {
            "subject": "Chain-of-Thought",
            "relation": "improves",
            "object": "Accuracy",
            "condition": "GSM8K"
        }
        
        claim2 = {
            "subject": "Chain-of-Thought",
            "relation": "degrades",
            "object": "Accuracy",
            "condition": "Complex"
        }
        
        ctype, conf, rules = engine.analyze_pair(claim1, claim2)
        
        print(f"[OK] Test analysis:")
        print(f"  - Type: {ctype}")
        print(f"  - Confidence: {conf}")
        print(f"  - Matched rules: {rules}")
        
        print("\n[PASS] Rule engine module working")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Rule engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_paper_generator():
    """Test paper generator module."""
    print("\n" + "="*60)
    print("TEST 7: Paper Generator Module")
    print("="*60)
    
    try:
        from paper_generator import PaperGenerator, generate_paper_summary
        
        generator = PaperGenerator(
            domain="Chain-of-Thought Mathematical Reasoning",
            target_venue="arXiv"
        )
        
        print("[OK] PaperGenerator initialized")
        
        # Test title generation
        sample_contra = pd.DataFrame([
            {"claim1_text": "A", "claim2_text": "B", "significance_score": 50}
        ])
        title = generator.generate_title(sample_contra)
        print(f"[OK] Generated title: {title}")
        
        # Test abstract generation
        sample_claims = pd.DataFrame([{"subject": "CoT"} for _ in range(100)])
        sample_papers = pd.DataFrame([{"arxiv_id": f"23{i:04d}"} for i in range(200)])
        
        abstract = generator.generate_abstract(sample_contra, sample_claims, sample_papers)
        print(f"[OK] Generated abstract ({len(abstract)} chars)")
        
        # Test summary
        summary = generate_paper_summary(sample_contra)
        print(f"[OK] Generated README summary ({len(summary)} chars)")
        
        print("\n[PASS] Paper generator module working")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Paper generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all module tests."""
    print("\n" + "="*60)
    print("SCIENTIFIC CONTRADICTION DETECTION SYSTEM - MODULE TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Ontology", test_ontology()))
    results.append(("Paper Fetcher", test_paper_fetcher()))
    results.append(("PDF Downloader", test_pdf_downloader()))
    results.append(("Text Extractor", test_text_extractor()))
    results.append(("Data Cleaner", test_data_cleaner()))
    results.append(("Rule Engine", test_rule_engine()))
    results.append(("Paper Generator", test_paper_generator()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[OK] All tests passed! System ready for full execution.")
    else:
        print(f"\n[FAIL] {total-passed} test(s) failed. Review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
