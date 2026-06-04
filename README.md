<p align="center">
  <img src="https://img.shields.io/badge/domain-CoT%20Math%20Reasoning-blue.svg"/>
  <img src="https://img.shields.io/badge/papers-200-blueviolet.svg"/>
  <img src="https://img.shields.io/badge/contradictions-165%20found-orange.svg"/>
  <img src="https://img.shields.io/badge/license-MIT-green.svg"/>
</p>

# Scientific Contradiction Detection System

**Automated detection of conflicting findings in scientific literature.**

This system scans hundreds of papers in a target domain, extracts structured claims, and surfaces where different papers reach opposing conclusions. It produces a **Contradiction Map** — a structured dataset of who disagrees with whom, about what, and why.

It is not a literature review tool. It is not a summarizer. It is a **disagreement detector** for the scientific record.

---

## The Problem

Academia produces far more papers than any researcher can read. In just five years (2021–2026), a narrow subfield like Chain-of-Thought mathematical reasoning accumulated ~200 papers. Conflicting conclusions are published in different venues, never cross-referenced. Researchers build on inconsistent foundations without knowing it.

The result is **Condition Drift**: papers use identical claim wording ("Chain-of-Thought improves accuracy") while testing on GPT-3.5 vs Llama-2 vs GPT-4, GSM8K vs MATH vs SVAMP — making the literature appear to agree when their experiments don't actually compare.

---

## Key Finding

In our production run, **76% of detected contradictions were not scientific disagreements** — they were Condition Drift. Only 24% were genuine logical contradictions between papers. This systematic inconsistency had not been previously documented at scale.

| Type | Count | Meaning |
|------|-------|---------|
| Experimental Condition | 126 (76%) | Same claim wording, different experimental setups |
| Logical Contradiction | 39 (24%) | Claims directly contradict each other |

---

## Pipeline

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

### Claim Format

Every claim in the system is a strict 6-tuple:

| subject | relation | object | condition | evidence | confidence |
|---------|----------|--------|-----------|----------|------------|
| Chain-of-Thought | improves | Arithmetic Accuracy | GSM8K, GPT-3.5 | Experimental_Result | 5 |

Contradictions are pairs of these tuples from different papers, classified by type and ranked by significance.

### Detection Strategies

Two complementary strategies find contradiction candidates:

**Strategy 1 — Same (subject, relation, object):** When two papers make the same claim but the claim is sensitive to experimental conditions (different models, datasets, parameters). These produce `Experimental_Condition_Contradiction`.

**Strategy 2 — Same (subject, object), opposite relations:** When one paper says "X improves Y" and another says "X degrades Y." These produce `Scientific_Dispute` or `Logical_Contradiction`.

A rule engine with 4 detection patterns handles confident cases; an LLM serves as fallback for the rest. Every contradiction gets a significance score: `0.6 × log(avg_citations) + 0.4 × avg_confidence`.

---

## Results

| Metric | Value |
|--------|-------|
| Papers collected | 200 |
| Papers with extractable claims | 39 |
| Structured claims extracted | 192 |
| Candidate contradiction pairs | 173 |
| **Verified contradictions** | **165** |
| Papers involved in contradictions | ~52 |

### Web Interface

![Contradiction Detection Dashboard](screenshots/homepage.png)

### Contradiction Network

![Contradiction Network](screenshots/contradiction_network.png)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your DeepSeek API key (choose one)
set DEEPSEEK_API_KEY=sk-your-key-here          # Windows CMD
$env:DEEPSEEK_API_KEY = "sk-your-key-here"     # PowerShell
export DEEPSEEK_API_KEY=sk-your-key-here       # Linux/macOS

# Run full pipeline
python src/main.py --phase all

# Or step by step
python src/main.py --phase 1          # Data collection
python src/main.py --phase 2 --mock   # Claim extraction (test with mock)
python src/main.py --phase 3          # Contradiction detection
python src/main.py --phase 4          # Paper generation

# Launch web interface
streamlit run src/app.py
```

### Individual Phase Scripts

```bash
python scripts/run_phase2.py          # Claim extraction via DeepSeek API
python scripts/run_phase2_batch.py    # Batch extraction (50 papers)
python scripts/run_phase3.py          # Contradiction detection
python scripts/run_phase3_final.py    # Final classification + paper gen
python scripts/run_phase4.py          # Paper generation only
python scripts/quick_test.py          # Quick 10-paper test
```

---

## Repository Structure

```
src/                          # Core pipeline modules
├── ontology.py               # Domain concepts, relations, synonym maps
├── paper_fetcher.py          # arXiv + Semantic Scholar data acquisition
├── pdf_downloader.py         # Rate-limited PDF download
├── text_extractor.py         # PyMuPDF text extraction
├── claim_extractor.py        # 6-tuple extraction (DeepSeek/Llama 3)
├── data_cleaner.py           # Ontology normalization + validation
├── batch_processor.py        # Extraction pipeline orchestrator
├── contradiction_detector.py # Candidate generation + classification
├── rule_engine.py            # 4 rule-based detection patterns
├── llm_client.py             # DeepSeek API client
├── paper_generator.py        # LaTeX paper generation
├── app.py                    # Streamlit web interface
└── main.py                   # CLI entry point

scripts/                      # Run scripts
├── run_full_pipeline.py      # Full pipeline
├── run_phase2.py / _batch.py # Claim extraction
├── run_phase3.py / _final.py # Contradiction detection
├── run_phase4.py             # Paper generation
├── fetch_relevant_papers.py  # Paper fetching
├── process_relevant_papers.py# PDF processing
├── check_and_clean.py        # Data cleaning
└── quick_test.py             # 10-paper test

tests/                        # Module tests
data/                         # Datasets (CSV)
screenshots/                  # Web interface screenshots
papers/                       # LaTeX paper source + PDF
templates/latex/              # LaTeX templates
docs/                         # Documentation
```

---

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
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

---

## Adapting to a New Domain

This system is domain-agnostic. To apply it to a different scientific field:

1. **Define ontology** — Edit `src/ontology.py`: replace `concepts`, `relations`, `synonym_map`
2. **Set search query** — Update `config.yaml` → `data.search_query`
3. **Run** — `python src/main.py --phase all`

Required: a formalizable vocabulary (10± concepts, 10± relations) and a searchable corpus of 150–500 papers.

---

## Methodology

This project follows the Hassabis Alpha-series scaffold approach:

1. **Narrow Domain** — Single formalized field (Chain-of-Thought math reasoning)
2. **Scaffold Development** — LLM for prototype (Phases 1–3), rule engine gradually replacing it
3. **Formal World Model** — Strict ontology validation before any AI processing
4. **Open Science** — Code and data publicly released

Non-goals: cross-domain analysis, paper quality evaluation, contradiction resolution, real-time monitoring, general extraction.

---

## Comparison

| Dimension | Traditional Review | This System |
|-----------|-------------------|-------------|
| Papers reviewed | 20–50 (manual) | 200+ (automated) |
| Claims representation | Mental model | Structured 6-tuple |
| Contradiction detection | Human memory | Rule engine + LLM |
| Impact weighting | Subjective | Citation × confidence |
| Reproducibility | None | Dataset + code published |
| Reusability | One-time | Re-runnable pipeline |

---

## Roadmap

- [x] Phase 1 — Data collection (ontology, fetch, download, extract)
- [x] Phase 2 — Claim extraction (6-tuple, normalization, batch processing)
- [x] Phase 3 — Contradiction detection (candidates, classification, scoring)
- [x] Phase 4 — Output (Streamlit dashboard, LaTeX paper, manual verification)
- [ ] Phase 5 — Intelligence layer (auto rule derivation, cross-domain, continuous monitoring)

---

## Citation

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

## Acknowledgments

- Methodology: Hassabis Alpha-series (AlphaGo, AlphaFold)
- LLM pipeline: Sakana AI Scientist
- Domain: Chain-of-Thought mathematical reasoning literature

---

> This system transforms scientific literature from a silent archive of disconnected findings into a **Contradiction Map** — making disagreement visible, measurable, and actionable.
