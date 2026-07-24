<div align="center">
<h2 align="center">
  <b>
    Scientific Contradiction Detection System
  </b>
</h2>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/domain-CoT%20Math%20Reasoning-blue.svg"/>
  <img src="https://img.shields.io/badge/papers-200-blueviolet.svg"/>
  <img src="https://img.shields.io/badge/contradictions-165%20found-orange.svg"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg"/>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &nbsp;|&nbsp;
  <a href="#overview">Overview</a> &nbsp;|&nbsp;
  <a href="#results">Results</a> &nbsp;|&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;|&nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="screenshots/homepage.png" width="800" />
</p>
<p align="center"><em>
  (a) Statistics Dashboard: contradiction metrics, type distribution, and significance scores.
  (b) Contradiction Network: paper-to-paper disagreement intensity matrix.
</em></p>

---

This is the official repository for **Scientific Contradiction Detection System**, an automated pipeline that scans hundreds of papers in a target domain, extracts structured claims, and surfaces where different papers reach opposing conclusions — producing a **Contradiction Map**: who disagrees with whom, about what, and why.

## Overview

Academic literature has a blind spot: conflicting conclusions are published in different venues and never cross-referenced. In Chain-of-Thought mathematical reasoning alone, ~200 papers were published between 2021–2026 with no systematic way to detect disagreements. Researchers build on inconsistent foundations without knowing it.

**The system makes disagreement measurable, structured, and queryable.**

Standard literature reviews rely on human memory to spot conflicts. This system replaces that with a four-phase automated pipeline that extracts structured 6-tuple claims and identifies contradictions via rule engine + LLM classification:

$$\text{200 papers} \to \text{192 structured claims} \to \text{165 verified contradictions}$$

### Core Data Structure

Every claim is a strict 6-tuple. Contradictions are pairs of tuples from different papers.

| subject | relation | object | condition | evidence | confidence |
|---------|----------|--------|-----------|----------|------------|
| Chain-of-Thought | improves | Arithmetic Accuracy | GSM8K, GPT-3.5 | Experimental_Result | 5 |

### Condition Drift

Papers state "Chain-of-Thought improves arithmetic accuracy" while testing on GPT-3.5 vs Llama-2 vs GPT-4, GSM8K vs MATH vs SVAMP. The literature *appears* to agree but their experiments don't actually compare. This systematic inconsistency had not been previously documented at scale.

<details>
<summary><b>Pipeline pseudocode</b></summary>

```python
# Phase 1: Data Collection
papers = fetch_arxiv(query="chain of thought mathematical reasoning", max=200)
pdfs = download_papers(papers, delay=15)
texts = extract_text(pdfs, sections=["Results", "Discussion", "Conclusion"])

# Phase 2: Claim Extraction
claims = []
for paper in texts:
    raw = llm_extract(paper, format="6-tuple")
    cleaned = normalize_ontology(raw, synonym_map, concept_list)
    claims.extend(cleaned)

# Phase 3: Contradiction Detection
candidates = generate_candidates(claims, same_sro=True, opposite_relations=True)
for pair in candidates:
    if rule_engine.match(pair):          # 4 deterministic patterns
        pair.type = rule_engine.classify(pair)
    else:
        pair.type = llm_classify(pair)   # LLM fallback
    pair.score = 0.6 * log(citations) + 0.4 * confidence

# Phase 4: Output
save_dashboard(contradictions)           # Streamlit
generate_latex_paper(contradictions)     # LaTeX
```

</details>

---

## Results

### Key Metrics

| Metric | Value |
|--------|-------|
| Papers collected | 200 |
| Papers with extractable claims | 39 |
| Structured claims extracted | 192 |
| Candidate contradiction pairs | 173 |
| **Verified contradictions** | **165** |
| Papers involved | ~52 |

### Contradiction Breakdown

| Contradiction Type | Count | Meaning |
|-------------------|-------|---------|
| Experimental Condition | 126 (76%) | Same claim wording, different experimental setups |
| Logical Contradiction | 39 (24%) | Claims directly contradict each other |

Only 24% are genuine scientific disputes. The rest reveal **Condition Drift** — a structural problem where papers appear to agree but their experiments don't actually compare.

### Screenshots

<p align="center">
  <img src="screenshots/contradiction_network.png" width="800" />
</p>
<p align="center"><em>Paper-Paper Contradiction Intensity Matrix — darker red indicates stronger disagreement.</em></p>

---

## Architecture

### Four-Phase Pipeline

```
200 papers
  │
  ▼ arXiv + Semantic Scholar
  ▼ PyMuPDF text extraction (Results/Discussion/Conclusion)
  │
  ▼ DeepSeek API / Llama 3 8B 4-bit
  ▼ 6-tuple claims: subject | relation | object | condition | evidence | confidence
  ▼ DataCleaner: ontology normalization + synonym mapping + confidence validation
  │
  ▼ 192 structured claims
  ▼ Candidate generation (same SRO + opposite relations)
  ▼ Rule engine (4 patterns) + LLM fallback classification
  ▼ Significance scoring: 0.6·log(citations) + 0.4·confidence
  │
  ▼ 165 verified contradictions
  ▼ Streamlit dashboard + LaTeX paper + manual verification
```

### Rule Engine (4 Patterns)

The rule engine handles classification without LLM inference, achieving zero hallucination for matched cases:

1. **Opposite relations** — Same subject/object, opposite relations → `Scientific_Dispute`
2. **Dataset conflict** — Same claim text, different datasets → `Experimental_Condition_Contradiction`
3. **Model comparison** — Same claim, different LLM models → `Experimental_Condition_Contradiction`
4. **Statistical disagreement** — Numerical values differ by >10% → `Statistical_Error_Contradiction`

### Significance Scoring

```python
significance = 0.6 * log1p(avg_citations) * 10 + 0.4 * avg_confidence * 10
```

Uses log-scaled citations to prevent a single high-citation paper from dominating the ranking.

### Modules

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Data** | `paper_fetcher.py` | arXiv + Semantic Scholar metadata retrieval |
| | `pdf_downloader.py` | Rate-limited PDF download (15s interval) |
| | `text_extractor.py` | PyMuPDF text from Results/Discussion/Conclusion |
| **Extraction** | `claim_extractor.py` | 6-tuple extraction via LLM (DeepSeek or Llama 3) |
| | `data_cleaner.py` | Ontology normalization, synonym mapping |
| | `batch_processor.py` | Orchestrates extraction across all papers |
| **Detection** | `contradiction_detector.py` | Candidate generation + classification + scoring |
| | `rule_engine.py` | 4 rule patterns + LLM fallback |
| | `llm_client.py` | DeepSeek API abstraction with retry logic |
| **Output** | `paper_generator.py` | LaTeX paper auto-generation |
| | `app.py` | Streamlit web interface (4 tabs) |
| **CLI** | `main.py` | Phase orchestration (`--phase 1..4`) |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your DeepSeek API key
set DEEPSEEK_API_KEY=sk-your-key-here          # Windows CMD
$env:DEEPSEEK_API_KEY = "sk-your-key-here"     # PowerShell
export DEEPSEEK_API_KEY=sk-your-key-here       # Linux/macOS

# Run full pipeline
python src/main.py --phase all

# Or step by step
python src/main.py --phase 1          # Data collection
python src/main.py --phase 2 --mock   # Claim extraction (test)
python src/main.py --phase 3          # Contradiction detection
python src/main.py --phase 4          # Paper generation

# Launch web interface
streamlit run src/app.py
```

<details>
<summary><b>Individual phase scripts</b></summary>

```bash
python scripts/run_phase2.py          # Claim extraction via API
python scripts/run_phase2_batch.py    # Batch extraction (50 papers)
python scripts/run_phase3.py          # Contradiction detection
python scripts/run_phase3_final.py    # Final classification + paper
python scripts/run_phase4.py          # Paper generation only
python scripts/quick_test.py          # 10-paper test run
```

</details>

### Configuration

```yaml
# config.yaml
data:
  search_query: "chain of thought mathematical reasoning"
  max_papers: 200
  start_year: 2021

model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  temperature: 0.05

scoring:
  citation_weight: 0.6
  confidence_weight: 0.4
```

### Adapting to a Different Domain

1. **Define ontology** — `src/ontology.py`: replace `concepts`, `relations`, `synonym_map`
2. **Set search query** — `config.yaml` → `data.search_query`
3. **Run** — `python src/main.py --phase all`

Requirements: a formalizable vocabulary (~10 concepts, ~10 relations), a searchable corpus of 150–500 papers.

---

## Citation

If you found our work useful, please cite

```bibtex
@software{contradiction_detector,
  title = {Scientific Contradiction Detection System},
  author = {Fengrru},
  year = {2026},
  url = {https://github.com/Fengrru/scientific-contradiction-detector}
}
```

## License

MIT License — Open source for academic and research use.

---

> This system transforms scientific literature from a silent archive of disconnected findings into a **Contradiction Map** — making disagreement visible, measurable, and actionable.
