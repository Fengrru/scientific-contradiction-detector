"""Quick test of DeepSeek API claim extraction on a single paper."""
import sys
sys.path.insert(0, "../src")

from llm_client import LLMClient
from claim_extractor import ClaimExtractor
from ontology import ONTOLOGY
import pandas as pd

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

client = LLMClient(api_key=API_KEY, model="deepseek-chat")
extractor = ClaimExtractor(api_client=client)

# Load a real paper text
df = pd.read_csv("data/papers_with_text.csv")
paper = df[df["text_extracted"] == True].iloc[0]
text = paper["full_text"][:8000]
print(f"Paper: {paper['arxiv_id']} - {paper['title'][:80]}")
print(f"Text length: {len(text)} chars")
print("Extracting claims via DeepSeek API...")
print("-" * 60)

claims = extractor.extract_from_text(text, ONTOLOGY.concepts, ONTOLOGY.relations)
print(f"\nExtracted {len(claims)} claims:")
for i, c in enumerate(claims):
    print(f"  [{i+1}] {c['subject']} | {c['relation']} | {c['object']}")
    print(f"       condition: {c['condition']}")
    print(f"       evidence: {c['evidence_type']}, confidence: {c['confidence']}")
