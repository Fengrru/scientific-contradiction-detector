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


def render_paper_analyzer():
    """Render single paper analysis interface."""
    st.header("📄 Paper Analyzer")
    
    st.markdown("Upload a paper to check for contradictions with existing literature.")
    
    uploaded_file = st.file_uploader(
        "Upload paper PDF or paste arXiv ID",
        type=["pdf"]
    )
    
    arxiv_id = st.text_input(
        "Or enter arXiv ID",
        placeholder="e.g., 2305.12345"
    )
    
    if st.button("Analyze", disabled=not (uploaded_file or arxiv_id)):
        st.info("Analysis feature requires full pipeline implementation.")
        st.markdown("""
        This would:
        1. Extract text from the paper
        2. Extract claims using the trained model
        3. Compare against the contradiction database
        4. Generate a contradiction report
        """)


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
