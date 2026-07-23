"""
Streamlit web application for Scientific Contradiction Detection System.
Interactive tool for browsing contradictions, statistics, and paper analysis.

Author: AI Scientist
Date: 2026-04-20
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ontology import ONTOLOGY
from data_cleaner import DataCleaner
from rule_engine import RuleEngine
from llm_client import LLMClient
from claim_extractor import ClaimExtractor
import fitz  # PyMuPDF for PDF text extraction
import re

st.set_page_config(
    page_title="Scientific Contradiction Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data():
    """Load contradiction data."""
    try:
        contradictions = pd.read_csv("./data/final_contradictions.csv")
        return contradictions
    except:
        return pd.DataFrame()


def load_claims():
    """Load claims data."""
    try:
        claims = pd.read_csv("./data/cleaned_claims.csv")
        return claims
    except:
        return pd.DataFrame()


def render_header():
    """Render application header."""
    st.title("🔍 Scientific Contradiction Detection System")
    st.markdown("""
    **Domain**: Chain-of-Thought Mathematical Reasoning in Large Language Models
    
    This tool automatically detects contradictions in scientific literature, 
    helping researchers identify conflicting findings and unresolved debates.
    """)


def render_sidebar():
    """Render sidebar filters."""
    with st.sidebar:
        st.header("Filters")
        
        # Contradiction type filter
        st.subheader("Contradiction Type")
        all_types = [
            "Experimental_Condition_Contradiction",
            "Measurement_Method_Contradiction",
            "Statistical_Error_Contradiction",
            "Scientific_Dispute",
            "Logical_Contradiction"
        ]
        selected_types = st.multiselect(
            "Select types to display",
            options=all_types,
            default=all_types
        )
        
        # Significance threshold
        st.subheader("Significance Threshold")
        min_significance = st.slider(
            "Minimum significance score",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=5.0
        )
        
        # Search
        st.subheader("Search")
        search_query = st.text_input(
            "Search by keyword",
            placeholder="e.g., GPT-4, GSM8K, accuracy"
        )
        
        return selected_types, min_significance, search_query


def render_statistics(contradictions_df):
    """Render statistics dashboard."""
    st.header("📊 Contradiction Statistics")
    
    if contradictions_df.empty:
        st.warning("No contradiction data available. Run the detection pipeline first.")
        return
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Contradictions", len(contradictions_df))
    
    with col2:
        avg_significance = contradictions_df["significance_score"].mean()
        st.metric("Average Significance", f"{avg_significance:.1f}")
    
    with col3:
        unique_papers = contradictions_df["claim1_arxiv"].nunique() + contradictions_df["claim2_arxiv"].nunique()
        st.metric("Papers Involved", unique_papers)
    
    with col4:
        high_impact = len(contradictions_df[contradictions_df["significance_score"] > 50])
        st.metric("High-Impact Contradictions", high_impact)
    
    # Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("By Contradiction Type")
        type_counts = contradictions_df["contradiction_type"].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            hole=0.4
        )
        st.plotly_chart(fig, width="stretch")
    
    with col_right:
        st.subheader("Significance Distribution")
        fig = px.histogram(
            contradictions_df,
            x="significance_score",
            nbins=20,
            labels={"significance_score": "Significance Score"}
        )
        st.plotly_chart(fig, width="stretch")


def render_contradiction_browser(contradictions_df, selected_types, min_significance, search_query):
    """Render contradiction browser."""
    st.header("📚 Contradiction Browser")
    
    if contradictions_df.empty:
        st.warning("No contradictions to display.")
        return
    
    # Apply filters
    filtered = contradictions_df[
        contradictions_df["contradiction_type"].isin(selected_types)
    ]
    
    filtered = filtered[filtered["significance_score"] >= min_significance]
    
    if search_query:
        search_lower = search_query.lower()
        filtered = filtered[
            filtered["claim1_text"].str.lower().str.contains(search_lower, na=False) |
            filtered["claim2_text"].str.lower().str.contains(search_lower, na=False) |
            filtered["claim1_condition"].str.lower().str.contains(search_lower, na=False) |
            filtered["claim2_condition"].str.lower().str.contains(search_lower, na=False)
        ]
    
    st.write(f"Showing {len(filtered)} contradictions")
    
    # Display contradictions
    for display_idx, (_, row) in enumerate(filtered.head(50).iterrows()):
        with st.expander(f"#{display_idx+1}: {row['claim1_text']} vs {row['claim2_text'][:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Claim 1**")
                st.write(f"📄 Paper: [{row['claim1_arxiv']}]")
                st.write(f"📝 Statement: {row['claim1_text']}")
                st.write(f"⚙️ Conditions: {row['claim1_condition']}")
                st.write(f"📊 Citations: {row['claim1_citations']}")
                st.write(f"✅ Confidence: {row['claim1_confidence']}/5")
            
            with col2:
                st.markdown("**Claim 2**")
                st.write(f"📄 Paper: [{row['claim2_arxiv']}]")
                st.write(f"📝 Statement: {row['claim2_text']}")
                st.write(f"⚙️ Conditions: {row['claim2_condition']}")
                st.write(f"📊 Citations: {row['claim2_citations']}")
                st.write(f"✅ Confidence: {row['claim2_confidence']}/5")
            
            st.divider()
            
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            
            with col_meta1:
                st.markdown(f"**Type**: {row['contradiction_type']}")
            
            with col_meta2:
                st.markdown(f"**Significance**: {row['significance_score']:.1f}")
            
            with col_meta3:
                if pd.notna(row.get("classification_reason")):
                    st.markdown(f"**Reason**: {row['classification_reason']}")


def render_network_view(contradictions_df):
    """Render network visualization of contradictions."""
    st.header("🕸️ Contradiction Network")
    
    if contradictions_df.empty or len(contradictions_df) < 2:
        st.info("Not enough data for network visualization (need 2+ contradictions)")
        return
    
    # Build network data
    nodes = set()
    edges = []
    
    for idx, row in contradictions_df.head(30).iterrows():
        nodes.add(row["claim1_arxiv"])
        nodes.add(row["claim2_arxiv"])
        edges.append({
            "source": row["claim1_arxiv"],
            "target": row["claim2_arxiv"],
            "significance": row["significance_score"],
            "type": row["contradiction_type"]
        })
    
    # Create network visualization using Plotly
    fig = go.Figure()
    
    # Simplified representation: heatmap of paper-paper contradictions
    papers = sorted(list(nodes))
    matrix = [[0 for _ in papers] for _ in papers]
    
    for edge in edges:
        if edge["source"] in papers and edge["target"] in papers:
            i = papers.index(edge["source"])
            j = papers.index(edge["target"])
            matrix[i][j] = edge["significance"]
            matrix[j][i] = edge["significance"]
    
    fig.add_trace(go.Heatmap(
        z=matrix,
        x=papers,
        y=papers,
        colorscale="Reds",
        showscale=True,
        name="Contradiction Strength"
    ))
    
    fig.update_layout(
        title="Paper-Paper Contradiction Intensity Matrix",
        xaxis_title="Paper arXiv ID",
        yaxis_title="Paper arXiv ID",
        height=600
    )
    
    st.plotly_chart(fig, width="stretch")


# ── Helper: extract text from uploaded PDF ──
def _extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"PDF text extraction failed: {e}")
        return ""


# ── Helper: keyword-based claim extraction (no LLM required) ──
def _extract_claims_keyword(text: str) -> pd.DataFrame:
    """Extract simple claims using ontology keyword matching.
    Works without any LLM — useful when no API key is available."""
    sentences = re.split(r'[.!?\n]+', text)
    
    # Build synonym → concept lookup
    synonym_to_concept = {}
    for orig, norm in ONTOLOGY.synonym_map.items():
        if norm in ONTOLOGY.concepts:
            synonym_to_concept[orig.lower()] = norm
    for c in ONTOLOGY.concepts:
        synonym_to_concept[c.lower()] = c
    
    # Sort by length descending to match longer (more specific) synonyms first
    synonym_items = sorted(synonym_to_concept.items(), key=lambda x: -len(x[0]))
    
    claims = []
    seen = set()
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30:
            continue
        sent_lower = sent.lower()
        
        # 1. Find subject (first ontology concept appearing in sentence)
        found_subject = None
        subj_pos = len(sent_lower)
        for synonym, concept in synonym_items:
            pos = sent_lower.find(synonym)
            if pos != -1 and pos < subj_pos:
                found_subject = concept
                subj_pos = pos
        
        if not found_subject:
            continue
        
        # 2. Find relation (ontology relation after the subject)
        found_relation = None
        rel_pos = len(sent_lower)
        for rel in ONTOLOGY.relations:
            pos = sent_lower.find(rel, subj_pos + len(found_subject))
            if pos != -1 and pos < rel_pos:
                found_relation = rel
                rel_pos = pos
        
        if not found_relation:
            continue
        
        # 3. Extract object (text after relation)
        obj_text = sent[rel_pos + len(found_relation):].strip().rstrip('.,;:')[:150]
        if not obj_text:
            continue
        
        # 4. Check for condition keywords in the sentence
        condition_kws = []
        for kw in ['gsm8k', 'math', 'svamp', 'gpt', 'llama', 'dataset', 'benchmark']:
            if kw in sent_lower:
                condition_kws.append(kw)
        condition = ', '.join(condition_kws) if condition_kws else ''
        
        # Deduplicate
        key = f"{found_subject}|{found_relation}|{obj_text[:60]}"
        if key not in seen:
            seen.add(key)
            claims.append({
                "subject": found_subject,
                "relation": found_relation,
                "object": obj_text,
                "condition": condition,
                "evidence_type": "Experimental_Result",
                "confidence": 3
            })
    
    return pd.DataFrame(claims)


# ── Helper: compare new claims against existing contradiction DB ──
def _compare_with_existing(new_claims: pd.DataFrame,
                            existing_claims: pd.DataFrame,
                            rule_engine: RuleEngine) -> pd.DataFrame:
    """Compare each new claim against the existing claim database.
    Uses the RuleEngine to detect contradictions."""
    if new_claims.empty or existing_claims.empty:
        return pd.DataFrame()
    
    results = []
    seen_pairs = set()
    
    for _, nc in new_claims.iterrows():
        nc_subj = str(nc.get('subject', ''))
        nc_rel = str(nc.get('relation', ''))
        nc_obj = str(nc.get('object', ''))
        nc_cond = str(nc.get('condition', ''))
        
        for _, ec in existing_claims.iterrows():
            ec_subj = str(ec.get('subject', ''))
            ec_rel = str(ec.get('relation', ''))
            ec_obj = str(ec.get('object', ''))
            ec_cond = str(ec.get('condition', ''))
            
            pair_key = tuple(sorted([f"{nc_subj}|{nc_rel}|{nc_obj}",
                                     f"{ec_subj}|{ec_rel}|{ec_obj}"]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            c1 = {"subject": nc_subj, "relation": nc_rel,
                   "object": nc_obj, "condition": nc_cond}
            c2 = {"subject": ec_subj, "relation": ec_rel,
                   "object": ec_obj, "condition": ec_cond}
            
            ctype, conf, rules = rule_engine.analyze_pair(c1, c2)
            if ctype != "No_Contradiction" and conf > 0:
                results.append({
                    "Your Claim": f"{nc_subj} {nc_rel} {nc_obj[:60]}",
                    "Existing Claim": f"{ec_subj} {ec_rel} {ec_obj[:60]}",
                    "Existing Paper": ec.get("arxiv_id", "N/A"),
                    "Contradiction Type": ctype,
                    "Rule Confidence": round(conf, 2),
                    "Matched Rules": ", ".join(rules)
                })
    
    return pd.DataFrame(results)


# ── Main Paper Analyzer UI ──
def render_paper_analyzer():
    """Render a functional Paper Analyzer — extract claims from a paper
    and check for contradictions with the existing literature database."""
    st.header("📄 Paper Analyzer")
    st.markdown("Upload a paper or paste its text to detect contradictions with the existing literature.")
    
    # ── Load existing data once ──
    existing_claims = load_claims()
    has_existing_data = not existing_claims.empty
    
    if not has_existing_data:
        st.warning("No existing claim database found. Run the pipeline first to build one.")
    
    rule_engine = RuleEngine() if has_existing_data else None
    
    # ── API key check ──
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    use_api = False
    llm_client = None
    if api_key:
        use_api = True
        llm_client = LLMClient(api_key=api_key)
    
    # ── Input modes ──
    input_mode = st.radio(
        "Input mode",
        ["Paste text", "Upload PDF"],
        horizontal=True
    )
    
    paper_text = ""
    
    if input_mode == "Paste text":
        paper_text = st.text_area(
            "Paste paper text (Results/Discussion/Conclusion sections):",
            height=300,
            placeholder="Paste the main findings of the paper here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload paper PDF",
            type=["pdf"],
            help="Upload a PDF to extract text from its Results/Discussion/Conclusion sections"
        )
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                paper_text = _extract_text_from_pdf(uploaded_file)
            if paper_text:
                st.success(f"Extracted {len(paper_text):,} characters from PDF")
                with st.expander("Preview extracted text"):
                    st.text(paper_text[:2000])
            else:
                st.error("Failed to extract text. Try pasting the text directly.")
    
    # ── Analyze ──
    col1, col2 = st.columns([3, 1])
    with col2:
        analyze_disabled = not paper_text or len(paper_text.strip()) < 50 or not has_existing_data
        analyze_btn = st.button("🔍 Analyze", disabled=analyze_disabled, type="primary")
    
    with col1:
        if use_api:
            st.info("✅ DeepSeek API detected — will use LLM extraction for better accuracy.")
        else:
            st.caption("ℹ️ No API key set — using keyword-based extraction (less accurate). "
                       "Set DEEPSEEK_API_KEY in .env for LLM extraction.")
    
    # ── Process ──
    if analyze_btn and paper_text and has_existing_data:
        # Step 1: Extract claims
        with st.spinner("Extracting claims..."):
            if use_api and llm_client:
                # Use LLM-based extraction
                extractor = ClaimExtractor(api_client=llm_client)
                raw_claims = extractor.extract_from_text(
                    paper_text[:12000],
                    concepts=ONTOLOGY.concepts,
                    relations=ONTOLOGY.relations
                )
                if raw_claims:
                    new_claims = pd.DataFrame(raw_claims)
                    cleaner = DataCleaner(ontology=ONTOLOGY)
                    new_claims = cleaner.process(new_claims)
                else:
                    new_claims = pd.DataFrame()
            else:
                # Use keyword-based extraction
                new_claims = _extract_claims_keyword(paper_text)
        
        if new_claims.empty:
            st.warning("No claims could be extracted from the provided text.")
            return
        
        # Show extracted claims
        st.subheader(f"📋 Extracted Claims ({len(new_claims)} found)")
        st.dataframe(
            new_claims[['subject', 'relation', 'object', 'condition']],
            use_container_width=True,
            hide_index=True
        )
        
        # Step 2: Compare with existing database
        with st.spinner("Checking for contradictions..."):
            contradictions = _compare_with_existing(new_claims, existing_claims, rule_engine)
        
        # Step 3: Show results
        if contradictions.empty:
            st.success("✅ No contradictions found with existing literature!")
        else:
            st.subheader(f"🚨 Potential Contradictions Detected ({len(contradictions)} found)")
            
            # Summary metrics
            type_counts = contradictions["Contradiction Type"].value_counts()
            cols = st.columns(len(type_counts))
            for i, (ctype, count) in enumerate(type_counts.items()):
                with cols[i]:
                    st.metric(ctype, count)
            
            # Detail table
            st.dataframe(contradictions, use_container_width=True, hide_index=True)
            
            # Download results
            csv = contradictions.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download contradiction report (CSV)",
                csv,
                "contradiction_report.csv",
                "text/csv"
            )


def main():
    """Main application entry point."""
    render_header()
    
    # Load data
    contradictions_df = load_data()
    
    # Sidebar filters
    selected_types, min_significance, search_query = render_sidebar()
    
    # Navigation
    tab_stats, tab_browser, tab_network, tab_analyzer = st.tabs([
        "📊 Statistics",
        "📚 Contradictions",
        "🕸️ Network",
        "📄 Analyzer"
    ])
    
    with tab_stats:
        render_statistics(contradictions_df)
    
    with tab_browser:
        render_contradiction_browser(
            contradictions_df,
            selected_types,
            min_significance,
            search_query
        )
    
    with tab_network:
        render_network_view(contradictions_df)
    
    with tab_analyzer:
        render_paper_analyzer()
    
    # Footer
    st.divider()
    st.caption("Scientific Contradiction Detection System | Built with Streamlit | Data: arXiv 2021-2026")


if __name__ == "__main__":
    main()
