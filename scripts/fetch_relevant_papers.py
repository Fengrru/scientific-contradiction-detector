"""
Fetch truly relevant papers about Chain-of-Thought math reasoning via Semantic Scholar API.
"""
import requests
import pandas as pd
import time
import os

# Semantic Scholar API (free, no key needed for basic usage)
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

queries = [
    "chain of thought prompting mathematical reasoning",
    "chain of thought large language model arithmetic",
    "CoT prompting improves accuracy GSM8K",
    "self-consistency chain of thought reasoning",
    "zero-shot chain of thought mathematical",
    "step by step reasoning large language model math",
    "few-shot chain of thought problem solving",
]

all_papers = {}
seen_ids = set()

for query in queries:
    print(f"\nSearching: '{query}'")
    params = {
        "query": query,
        "limit": 30,
        "fields": "title,abstract,year,authors,citationCount,externalIds,publicationDate"
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for p in data.get("data", []):
                arxiv_id = None
                if p.get("externalIds") and p["externalIds"].get("ArXiv"):
                    arxiv_id = p["externalIds"]["ArXiv"]
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    all_papers[arxiv_id] = {
                        "arxiv_id": arxiv_id,
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "year": p.get("year", 0),
                        "citations": p.get("citationCount", 0),
                        "authors": [a["name"] for a in p.get("authors", [])],
                    }
            print(f"  Found {len(data.get('data',[]))} results, {len(seen_ids)} unique so far")
        else:
            print(f"  Error: {resp.status_code}")
    except Exception as e:
        print(f"  Failed: {e}")
    time.sleep(1.5)  # Rate limit

print(f"\n{'=' * 60}")
print(f"Total unique papers: {len(all_papers)}")
print(f"{'=' * 60}")

if all_papers:
    df = pd.DataFrame(list(all_papers.values()))
    df = df.sort_values("citations", ascending=False).reset_index(drop=True)
    
    # Add pdf_url
    df["pdf_url"] = df["arxiv_id"].apply(lambda x: f"https://arxiv.org/pdf/{x}.pdf")
    df["published"] = ""
    
    output_path = "data/relevant_core_papers.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} papers to {output_path}")
    
    print("\nTop 10 by citations:")
    for _, row in df.head(10).iterrows():
        print(f"  [{row['citations']:4d}] {row['arxiv_id']}: {row['title'][:70]}")
else:
    print("No papers found!")
