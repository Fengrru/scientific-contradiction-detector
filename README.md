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

## Why This Exists

Academia is drowning in its own output. A narrow subfield like Chain-of-Thought mathematical reasoning produced ~200 papers between 2021 and 2026 — more than any researcher can read. Conflicting conclusions are published in different venues and never cross-referenced. Researchers build on inconsistent foundations without knowing it.

| Problem | Consequence |
|---------|-------------|
| Volume overload | ~200 papers in 5 years — impossible to read manually |
| Conclusion fragmentation | Conflicting findings never compared |
| Condition blindness | Identical wording hides different experimental setups |
| No audit trail | No systematic record of where evidence disagrees |

This system solves the last three problems by making disagreement **measurable, structured, and queryable**.

---

## Running Results

In a production run on Chain-of-Thought Mathematical Reasoning literature (2021–2026), 200 papers were collected and analyzed. The results:

| Metric | Value |
|--------|-------|
| Papers collected | 200 |
| Papers with extracted claims | 39 |
| Structured claims extracted | 192 |
| Candidate contradiction pairs | 173 |
| **Verified contradictions** | **165** |

**Breakdown by type:**
| Type | Count | Meaning |
|------|-------|---------|
| Experimental Condition | 126 (76%) | Same claim wording, different experimental setups |
| Logical Contradiction | 39 (24%) | Claims directly contradict each other |

**Key finding:** 76% of contradictions are not scientific disagreements — they are *Condition Drift*. Papers state "Chain-of-Thought improves arithmetic accuracy" while testing on GPT-3.5 vs Llama-2 vs GPT-4, GSM8K vs MATH vs SVAMP. The literature *appears* to agree but their experiments don't actually compare. This systematic inconsistency had not been previously documented at scale.

---

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│                   1. DATA COLLECTION                      │
│                                                          │
│   arXiv API ──► paper metadata (200 papers)              │
│   Semantic Scholar ──► citation counts                   │
│   PDF download (15s rate limit)                          │
│   PyMuPDF text extraction (Results/Discussion/Conclusion)│
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                   2. CLAIM EXTRACTION                     │
│                                                          │
│   DeepSeek API / Llama 3 8B 4-bit                        │
│   ──► 6-tuple: <subject> | <relation> | <object>        │
│                | <condition> | <evidence> | <confidence> │
│                                                          │
│   DataCleaner:                                           │
│   ├── Ontology normalization (GPT-4 → Large Language Model)│
│   ├── Synonym mapping (enhances → improves)               │
│   └── Confidence validation (1-5 scale)                  │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                   3. CONTRADICTION DETECTION               │
│                                                          │
│   Strategy 1: Same (subject, relation, object)           │
│     ──► Experimental condition contradictions            │
│   Strategy 2: Opposite relations                         │
│     ──► Scientific disputes                              │
│                                                          │
│   Classification: Rule engine (4 patterns) + LLM fallback│
│   Significance score: 0.6·log(citations) + 0.4·confidence│
│   ──► 165 contradictions ranked by impact                │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                   4. INTERFACE & OUTPUT                   │
│                                                          │
│   Streamlit web app (browse, filter, visualize)           │
│   LaTeX research paper (auto-generated)                   │
│   Manual verification sample (top 100)                    │
│   Rule engine from verified patterns                     │
└──────────────────────────────────────────────────────────┘
```

### Core Data Structure

Every claim in the system is a strict 6-tuple:

| subject | relation | object | condition | evidence | confidence |
|---------|----------|--------|-----------|----------|------------|
| Chain-of-Thought | improves | Arithmetic Accuracy | GSM8K, GPT-3.5 | Experimental_Result | 5 |

Contradictions are pairs of these tuples from different papers, classified by type and ranked by significance.

### Web Interface

![Contradiction Detection Dashboard](screenshots/homepage.png)

### Contradiction Network

![Contradiction Network](screenshots/contradiction_network.png)

---

## Comparison

| Dimension | Traditional Approach | This System |
|-----------|---------------------|-------------|
| Papers reviewed | 20–50 (manual) | 200+ (automated) |
| Claims representation | Mental model | Structured 6-tuple |
| Contradiction detection | Human memory | Rule engine + LLM |
| Impact weighting | Subjective | Citation × confidence |
| Reproducibility | None | Dataset + code published |
| Frequency | One-time publication | Re-runnable pipeline |

---

## Non-Goals

This system currently does not attempt to provide:

- **Cross-domain analysis** — operates within a single formalized domain
- **Paper quality evaluation** — does not judge methodology, only surfaces conflicting claims
- **Resolution or adjudication** — identifies disagreements but does not resolve them
- **Real-time monitoring** — designed for batch analysis, not live feeds
- **General extraction** — ontology is domain-specific by design
- **Windows local LLM** — 4-bit quantization via `bitsandbytes` is Linux/macOS only; Windows users should use the DeepSeek API path (`scripts/run_full_pipeline.py`)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your DeepSeek API key (choose one)
set DEEPSEEK_API_KEY=sk-your-key-here          # Windows CMD
$env:DEEPSEEK_API_KEY = "sk-your-key-here"     # PowerShell
export DEEPSEEK_API_KEY=sk-your-key-here        # Linux/macOS

# Run everything
python src/main.py --phase all

# Or step by step
python src/main.py --phase 1          # Data collection
python src/main.py --phase 2 --mock   # Claim extraction (test)
python src/main.py --phase 3          # Contradiction detection
python src/main.py --phase 4          # Paper generation

# Web interface
streamlit run src/app.py
```

---

## Repository Structure

```
src/                          # Pipeline modules
├── ontology.py               # Domain concepts, relations, contradiction types
├── paper_fetcher.py          # arXiv + Semantic Scholar data acquisition
├── pdf_downloader.py         # Rate-limited PDF download
├── text_extractor.py         # PDF text extraction (PyMuPDF)
├── claim_extractor.py        # 6-tuple claim extraction (DeepSeek/Llama)
├── data_cleaner.py           # Ontology normalization
├── batch_processor.py        # Extraction pipeline orchestrator
├── contradiction_detector.py # Candidate generation, classification, scoring
├── rule_engine.py            # Rule-based contradiction classification
├── llm_client.py             # DeepSeek API client
├── paper_generator.py        # LaTeX paper generation
├── app.py                    # Streamlit web interface
└── main.py                   # CLI entry point

scripts/                      # Run scripts & utilities
├── run_full_pipeline.py      # Full pipeline runner
├── run_phase2.py             # Phase 2: claim extraction
├── run_phase2_batch.py       # Phase 2: batch extraction
├── run_phase3.py             # Phase 3: contradiction detection
├── run_phase3_final.py       # Phase 3: final classification
├── run_phase4.py             # Phase 4: paper generation
├── pipeline_full_rerun.py    # Full pipeline re-run
├── fetch_relevant_papers.py  # Fetch papers via API
├── process_relevant_papers.py# Process downloaded papers
├── check_and_clean.py        # Data cleaning utility
└── quick_test.py             # Quick pipeline test

tests/                        # Module tests
├── test_modules.py           # Component integration tests
└── test_api_extraction.py    # DeepSeek API extraction test

data/                         # Datasets (CSV)
screenshots/                  # README screenshots
papers/                       # LaTeX paper source & PDFs
templates/latex/              # LaTeX templates
docs/                         # Documentation
│   └── plan.md               # Project plan (Chinese)
```

---

## Roadmap

**Phase 1 — Data Layer** (Complete)
- [x] Ontology definition
- [x] Paper metadata acquisition
- [x] PDF download with rate limiting
- [x] Text extraction from core sections

**Phase 2 — Extraction Layer** (Complete)
- [x] 6-tuple claim extraction
- [x] Ontology normalization
- [x] Batch processing pipeline

**Phase 3 — Detection Layer** (Complete)
- [x] Candidate pair generation
- [x] Contradiction type classification
- [x] Significance scoring
- [x] Rule engine (4 detection patterns)
- [x] Manual verification sampling

**Phase 4 — Output Layer** (Complete)
- [x] Streamlit interactive dashboard
- [x] Contradiction browser and filters
- [x] LaTeX paper generation

**Next — Intelligence Layer** (Planned)
- [ ] Automated rule derivation from verified contradictions
- [ ] Cross-domain expansion
- [ ] Continuous arXiv monitoring
- [ ] Full rule engine coverage (>90%)

---

## Configuration

Edit `config.yaml` at the project root to customize:

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

## Methodology

This project follows the Hassabis Alpha-series scaffold approach:

1. **Narrow Domain** — Focused on Chain-of-Thought mathematical reasoning only
2. **Scaffold Development** — LLM for prototype (Phases 1-3), rule engine replaces 80% (Phase 9)
3. **Formal World Model** — Defined ontology with strict validation before AI processing
4. **Open Science** — All code and data publicly released

---

## Role in the AI Scientist Pipeline

This system is designed as a core component of an **automated AI for Science workflow** — a closed loop where AI reads papers, identifies disagreements, generates hypotheses, and designs experiments to resolve them.

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Scientist Pipeline                       │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Paper    │───►│ Contradiction │───►│ Hypothesis      │   │
│  │ Fetcher  │    │ Detector      │    │ Generator       │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                       │                         │            │
│                       ▼                         ▼            │
│                ┌──────────────┐    ┌──────────────────┐      │
│                │ Contradiction│    │ Experiment      │      │
│                │ Map          │    │ Designer        │      │
│                └──────────────┘    └──────────────────┘      │
│                                            │                  │
│                                            ▼                  │
│                                  ┌──────────────────┐         │
│                                  │ Paper Writer     │         │
│                                  │ (LaTeX Gen)      │         │
│                                  └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**This project occupies the detection layer** — it scans the literature, extracts structured claims, and produces the Contradiction Map that feeds into hypothesis generation. Without this step, an AI Scientist has no systematic way to know where the evidence actually disagrees.

The full vision (inspired by Sakana AI Scientist and the Hassabis Alpha-series):
1. **Read** — Fetch and extract claims from papers (this project, Phase 1-2)
2. **Detect** — Find contradictions and map disagreements (this project, Phase 3)
3. **Hypothesize** — Generate testable hypotheses to resolve contradictions (future work)
4. **Experiment** — Design and run computational experiments (future work)
5. **Write** — Auto-generate papers describing findings (this project, Phase 4)

This pipeline is domain-agnostic. By swapping the ontology in `src/ontology.py` and the search query in `config.yaml`, the same system can map contradictions in any scientific field with a formalizable vocabulary.

---

## Citation

```bibtex
@software{contradiction_detector,
  title = {Scientific Contradiction Detection System},
  author = {AI Scientist},
  year = {2026},
  url = {https://github.com/Fengrru/scientific-contradiction-detector}
}
```

## License

MIT License — Open source for academic and research use.

## Acknowledgments

- Methodology based on Hassabis Alpha-series (AlphaGo, AlphaFold)
- LLM pipeline inspired by Sakana AI Scientist
- Domain expertise: Chain-of-Thought mathematical reasoning literature

---

> This system transforms scientific literature from a silent archive of disconnected findings into a **Contradiction Map** — making disagreement visible, measurable, and actionable for the first time.
