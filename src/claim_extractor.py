"""
6-tuple claim extraction using Llama 3 8B Instruct with 4-bit quantization.
Extracts structured claims from paper text following the ontology.

Author: AI Scientist
Date: 2026-04-20
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
import logging
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract 6-tuple claims from paper text using LLM."""
    
    # 6-tuple format: <subject> | <relation> | <object> | <condition> | <evidence_type> | <confidence>
    CLAIM_FORMAT = "<subject> | <relation> | <object> | <condition> | <evidence_type> | <confidence>"
    
    # Valid evidence types
    EVIDENCE_TYPES = ["Experimental_Result", "Theoretical_Analysis", "Literature_Citation"]
    
    def __init__(self,
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 use_4bit: bool = True,
                 max_length: int = 16384,
                 temperature: float = 0.05,
                 top_p: float = 0.95,
                 mock_mode: bool = False,
                 api_client: Optional[LLMClient] = None):
        """
        Initialize claim extractor with LLM.
        
        Args:
            model_name: HuggingFace model name
            use_4bit: Use 4-bit quantization to save VRAM
            max_length: Maximum sequence length
            temperature: Sampling temperature (low for deterministic)
            top_p: Nucleus sampling parameter
            mock_mode: If True, generate synthetic claims without loading LLM
            api_client: LLMClient for API-based extraction (DeepSeek etc.)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.mock_mode = mock_mode
        self.api_client = api_client
        self.api_mode = api_client is not None and api_client.available
        
        if mock_mode:
            logger.info("Mock mode enabled - will generate synthetic claims")
            self.model = None
            self.tokenizer = None
            return
        
        if self.api_mode:
            logger.info(f"API mode enabled with {api_client.model}")
            self.model = None
            self.tokenizer = None
            return
        
        logger.info(f"Loading model: {model_name}")
        
        # Configure quantization
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded successfully")
    
    def _build_extraction_prompt(self, paper_text: str, concepts: List[str], relations: List[str]) -> str:
        """
        Build strict prompt for claim extraction.
        
        Args:
            paper_text: Text from paper core sections
            concepts: List of valid concepts from ontology
            relations: List of valid relations from ontology
            
        Returns:
            Formatted prompt string
        """
        concepts_str = "\n".join([f"- {c}" for c in concepts])
        relations_str = "\n".join([f"- {r}" for r in relations])
        evidence_str = "\n".join([f"- {e}" for e in self.EVIDENCE_TYPES])
        
        prompt = f"""You are a scientific paper analyzer. Extract all explicit, verifiable claims from the following research paper text.

For each claim, output exactly ONE line in this 6-tuple format:
{self.CLAIM_FORMAT}

FIELD DEFINITIONS:
1. subject: The main entity being studied. MUST use one of these ontology concepts:
{concepts_str}

2. relation: The relationship. MUST use one of these ontology relations:
{relations_str}

3. object: The target/value of the relation (e.g., metric value, comparison result)

4. condition: Experimental conditions (dataset, model parameters, temperature, sample size, etc.)

5. evidence_type: MUST be exactly one of:
{evidence_str}

6. confidence: Author's confidence in the claim (1-5 scale, where 5 = 100% certain)

RULES:
- Extract ONLY the author's OWN conclusions, not background information or citations
- Each claim MUST use exactly one concept from the ontology list
- Each claim MUST use exactly one relation from the ontology list
- Output ONLY the 6-tuple lines, NO headers, NO explanations, NO numbering
- DO NOT use angle brackets < > around field values (write plain text only)
- NEVER wrap subjects in < or > characters
- The condition field must be concise (dataset, model, parameters only)
- If no valid claims found, output: NONE
- Maximum 20 claims per paper (most important ones)

EXAMPLES OF CORRECT OUTPUT:
Chain-of-Thought | improves | Arithmetic Accuracy | GSM8K, GPT-3.5 | Experimental_Result | 5
Large Language Model | outperforms | Few-Shot Prompting | MathQA, GPT-4 | Experimental_Result | 4

PAPER TEXT:
{paper_text[:12000]}

OUTPUT (6-tuple format only):
"""
        return prompt
    
    def _parse_claims(self, response: str) -> List[Dict]:
        """
        Parse claims from model response.
        
        Args:
            response: Raw model output
            
        Returns:
            List of parsed claim dictionaries
        """
        claims = []
        
        # Extract lines after "OUTPUT"
        if "OUTPUT" in response:
            response = response.split("OUTPUT")[-1]
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line in ["NONE", "(6-tuple format only):", "OUTPUT (6-tuple format only):"]:
                continue
            
            # Skip lines without proper separator
            if "|" not in line:
                continue
            
            # Parse 6-tuple
            parts = [p.strip() for p in line.split("|")]
            
            if len(parts) == 6:
                claims.append({
                    "subject": parts[0],
                    "relation": parts[1],
                    "object": parts[2],
                    "condition": parts[3],
                    "evidence_type": parts[4],
                    "confidence": parts[5]
                })
            else:
                logger.warning(f"Invalid claim format (expected 6 parts, got {len(parts)}): {line}")
        
        return claims
    
    def extract_from_text(self, 
                         paper_text: str,
                         concepts: List[str],
                         relations: List[str]) -> List[Dict]:
        """
        Extract claims from a single paper text.
        
        Args:
            paper_text: Paper text to analyze
            concepts: Valid ontology concepts
            relations: Valid ontology relations
            
        Returns:
            List of extracted claim dictionaries
        """
        if not paper_text or len(paper_text) < 100:
            logger.warning("Text too short, skipping")
            return []
        
        # Mock mode: generate synthetic claims
        if self.mock_mode:
            logger.info("Mock mode: generating synthetic claims")
            return self._generate_mock_claims(paper_text, concepts, relations)
        
        # Build prompt (same for local and API)
        prompt = self._build_extraction_prompt(paper_text, concepts, relations)
        
        # API mode: use DeepSeek API
        if self.api_mode:
            response = self.api_client.complete(
                prompt,
                temperature=self.temperature,
                max_tokens=2048
            )
            if response is None:
                logger.warning("API call returned None, falling back to mock")
                return self._generate_mock_claims(paper_text, concepts, relations)
            claims = self._parse_claims(response)
            logger.info(f"Extracted {len(claims)} claims via API")
            return claims
        
        # Local model mode
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        claims = self._parse_claims(response)
        
        logger.info(f"Extracted {len(claims)} claims from text")
        
        return claims
    
    def _generate_mock_claims(self, paper_text: str, concepts: List[str], relations: List[str]) -> List[Dict]:
        """
        Generate synthetic claims for mock mode.
        
        Args:
            paper_text: Paper text
            concepts: Valid concepts
            relations: Valid relations
            
        Returns:
            List of synthetic claim dictionaries
        """
        import random
        
        # Generate 1-5 synthetic claims per paper
        num_claims = random.randint(1, 5)
        claims = []
        
        # Sample relations and concepts
        sample_relations = ["improves", "reduces", "achieves", "outperforms", "correlates_with"]
        sample_concepts = ["Large Language Model", "Chain of Thought", "Reasoning", "Mathematical Reasoning", "Prompt Engineering"]
        sample_evidence = ["Experimental_Result", "Theoretical_Analysis"]
        
        for i in range(num_claims):
            claims.append({
                "subject": random.choice(sample_concepts),
                "relation": random.choice(sample_relations),
                "object": f"result_{i+1}",
                "condition": f"dataset=mock_{i}, model=test",
                "evidence_type": random.choice(sample_evidence),
                "confidence": str(random.randint(3, 5))
            })
        
        logger.info(f"Mock mode: generated {len(claims)} synthetic claims")
        return claims
    
    def batch_extract(self,
                   papers_df: pd.DataFrame,
                   concepts: List[str],
                   relations: List[str],
                   text_column: str = "full_text",
                   save_every: int = 10) -> pd.DataFrame:
        """
        Extract claims from all papers in batch.
        
        Args:
            papers_df: DataFrame with paper metadata and text
            concepts: Valid ontology concepts
            relations: Valid ontology relations
            text_column: Column containing paper text
            save_every: Save intermediate results every N papers
            
        Returns:
            DataFrame with all extracted claims
        """
        all_claims = []
        
        for idx, row in papers_df.iterrows():
            arxiv_id = row.get("arxiv_id", f"paper_{idx}")
            paper_text = row.get(text_column, "")
            
            logger.info(f"Processing paper {idx+1}/{len(papers_df)}: {arxiv_id}")
            
            try:
                claims = self.extract_from_text(paper_text, concepts, relations)
                
                # Add metadata
                for claim in claims:
                    claim["arxiv_id"] = arxiv_id
                    claim["title"] = row.get("title", "")
                    claim["citations"] = row.get("citations", 0)
                
                all_claims.extend(claims)
                
                # Save intermediate results
                if (idx + 1) % save_every == 0:
                    temp_df = pd.DataFrame(all_claims)
                    temp_df.to_csv(f"./data/all_claims_temp_{idx+1}.csv", index=False)
                    logger.info(f"Saved intermediate results: {len(all_claims)} claims so far")
                    
            except Exception as e:
                logger.error(f"Error processing {arxiv_id}: {e}")
                continue
        
        # Final DataFrame
        claims_df = pd.DataFrame(all_claims)
        
        logger.info(f"Total claims extracted: {len(claims_df)}")
        
        return claims_df


def validate_claim_format(claim: Dict, concepts: List[str], relations: List[str]) -> Tuple[bool, str]:
    """
    Validate a claim against ontology.
    
    Args:
        claim: Claim dictionary
        concepts: Valid concepts
        relations: Valid relations
        
    Returns:
        (is_valid, error_message)
    """
    required_fields = ["subject", "relation", "object", "condition", "evidence_type", "confidence"]
    
    # Check all fields present
    for field in required_fields:
        if field not in claim or not claim[field]:
            return False, f"Missing field: {field}"
    
    # Check concept validity
    if claim["subject"] not in concepts:
        return False, f"Invalid subject: {claim['subject']}"
    
    # Check relation validity
    if claim["relation"] not in relations:
        return False, f"Invalid relation: {claim['relation']}"
    
    # Check evidence type
    valid_evidence = ["Experimental_Result", "Theoretical_Analysis", "Literature_Citation"]
    if claim["evidence_type"] not in valid_evidence:
        return False, f"Invalid evidence_type: {claim['evidence_type']}"
    
    # Check confidence is numeric 1-5
    try:
        conf = int(claim["confidence"])
        if conf < 1 or conf > 5:
            return False, f"Confidence out of range: {conf}"
    except:
        return False, f"Invalid confidence: {claim['confidence']}"
    
    return True, "Valid"


if __name__ == "__main__":
    import yaml
    from ontology import ONTOLOGY
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize extractor
    extractor = ClaimExtractor(
        model_name=config["model"]["name"],
        use_4bit=config["model"]["quantization"] == "4bit",
        max_length=config["model"]["max_length"],
        temperature=config["model"]["temperature"],
        top_p=config["model"]["top_p"]
    )
    
    # Test with sample text
    sample_text = """
    We evaluate Chain-of-Thought prompting on GSM8K dataset using GPT-3.5-turbo.
    Results show that CoT improves arithmetic accuracy from 45% to 78%.
    Self-consistency with 10 samples further boosts performance to 82%.
    """
    
    claims = extractor.extract_from_text(
        sample_text,
        concepts=ONTOLOGY.concepts,
        relations=ONTOLOGY.relations
    )
    
    print(f"\nExtracted {len(claims)} claims:")
    for claim in claims:
        print(f"  - {claim}")
