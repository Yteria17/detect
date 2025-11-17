"""Streamlit dashboard for misinformation detection system."""

import streamlit as st
import requests
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Misinformation Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verdict-supported {
        color: #28a745;
        font-weight: bold;
    }
    .verdict-refuted {
        color: #dc3545;
        font-weight: bold;
    }
    .verdict-insufficient {
        color: #ffc107;
        font-weight: bold;
    }
    .verdict-conflicting {
        color: #ff8800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def submit_claim(claim: str, priority: str = "normal") -> Optional[Dict]:
    """Submit a claim for fact-checking."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/fact-check",
            json={"claim": claim, "priority": priority},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error submitting claim: {str(e)}")
        return None


def get_claim_result(claim_id: str) -> Optional[Dict]:
    """Get fact-check result by claim ID."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/fact-check/{claim_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error retrieving result: {str(e)}")
        return None


def get_statistics() -> Optional[Dict]:
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def display_verdict(verdict: str, confidence: float):
    """Display verdict with styling."""
    verdict_class = {
        "SUPPORTED": "verdict-supported",
        "REFUTED": "verdict-refuted",
        "INSUFFICIENT_INFO": "verdict-insufficient",
        "CONFLICTING": "verdict-conflicting"
    }.get(verdict, "")

    emoji = {
        "SUPPORTED": "✅",
        "REFUTED": "❌",
        "INSUFFICIENT_INFO": "❓",
        "CONFLICTING": "⚠️"
    }.get(verdict, "")

    st.markdown(
        f'<p class="{verdict_class}" style="font-size: 1.5rem;">{emoji} {verdict}</p>',
        unsafe_allow_html=True
    )
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.1%}")


# Sidebar
with st.sidebar:
    st.title("🔍 Navigation")

    # API Health Check
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")

    page = st.radio(
        "Select Page",
        ["Fact Check", "Recent Checks", "Statistics", "About"]
    )

    st.markdown("---")

    # System info
    st.caption("**System Information**")
    st.caption(f"API Endpoint: {API_BASE_URL}")
    st.caption(f"Version: 1.0.0")


# Main content
st.markdown('<p class="main-header">🔍 Misinformation Detection System</p>', unsafe_allow_html=True)

if page == "Fact Check":
    st.header("Submit a Claim for Verification")

    # Input form
    with st.form("claim_form"):
        claim_text = st.text_area(
            "Enter the claim to verify:",
            height=150,
            placeholder="e.g., 'The president announced new policies yesterday...'"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            priority = st.selectbox(
                "Priority",
                ["normal", "high", "urgent"],
                index=0
            )
        with col2:
            submit_button = st.form_submit_button("🔍 Verify Claim", use_container_width=True)

    if submit_button and claim_text:
        if not api_healthy:
            st.error("API is not available. Please check the connection.")
        else:
            with st.spinner("Analyzing claim... This may take a moment."):
                result = submit_claim(claim_text, priority)

                if result:
                    st.success("✅ Analysis Complete!")

                    # Display results
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("Verdict")
                        display_verdict(result['verdict'], result['confidence'])

                    with col2:
                        st.subheader("Metadata")
                        st.caption(f"**Claim ID:** {result['claim_id']}")
                        st.caption(f"**Analyzed:** {result['created_at']}")

                    # Detailed results
                    st.markdown("---")

                    full_result = get_claim_result(result['claim_id'])

                    if full_result:
                        # Tabs for different views
                        tab1, tab2, tab3, tab4 = st.tabs(
                            ["📊 Summary", "🔬 Assertions", "📰 Evidence", "🔄 Processing"]
                        )

                        with tab1:
                            st.subheader("Classification")
                            if full_result.get('classification'):
                                cls = full_result['classification']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Theme", cls.get('theme', 'N/A'))
                                with col2:
                                    st.metric("Complexity", f"{cls.get('complexity', 0)}/10")
                                with col3:
                                    st.metric("Urgency", f"{cls.get('urgency', 0)}/10")

                            # Anomaly detection
                            if full_result.get('anomaly_detection'):
                                st.subheader("Anomaly Detection")
                                anom = full_result['anomaly_detection']
                                st.metric(
                                    "High Anomaly Assertions",
                                    anom.get('high_anomaly_count', 0)
                                )
                                st.metric(
                                    "Average Anomaly Score",
                                    f"{anom.get('average_anomaly_score', 0):.2f}"
                                )

                        with tab2:
                            st.subheader("Assertion Analysis")
                            assertions = full_result.get('assertions', [])

                            for i, assertion in enumerate(assertions, 1):
                                with st.expander(f"Assertion {i}: {assertion['text'][:100]}..."):
                                    st.write(f"**Full text:** {assertion['text']}")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Verdict", assertion['verdict'])
                                    with col2:
                                        st.metric("Confidence", f"{assertion['confidence']:.1%}")
                                    st.caption(f"Evidence sources: {assertion['evidence_count']}")

                        with tab3:
                            st.subheader("Evidence Sources")
                            evidence_summary = full_result.get('evidence_summary', {})

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Sources", evidence_summary.get('total_sources', 0))
                            with col2:
                                st.metric(
                                    "High Credibility",
                                    evidence_summary.get('high_credibility_sources', 0)
                                )
                            with col3:
                                st.metric(
                                    "Unique Domains",
                                    evidence_summary.get('unique_domains', 0)
                                )

                            # Show evidence if available
                            if result.get('evidence_summary'):
                                st.subheader("Top Evidence Sources")
                                for ev in result['evidence_summary'][:10]:
                                    st.markdown(
                                        f"- **{ev['domain']}** (credibility: {ev['credibility']:.0%}) - [{ev['url']}]({ev['url']})"
                                    )

                        with tab4:
                            st.subheader("Processing Timeline")
                            processing = full_result.get('processing', {})

                            st.caption(f"**Agents Involved:** {', '.join(processing.get('agents_involved', []))}")
                            st.caption(f"**Processing Time:** {processing.get('processing_time_ms', 0)} ms")

                            # Reasoning trace
                            if result.get('reasoning_trace'):
                                st.subheader("Reasoning Trace")
                                for step in result['reasoning_trace']:
                                    st.caption(f"• {step}")

elif page == "Recent Checks":
    st.header("Recent Fact Checks")

    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/fact-check?limit=20", timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])

            if results:
                for result in results:
                    with st.expander(
                        f"{result['verdict']} - {result['claim'][:100]}...",
                        expanded=False
                    ):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Claim:** {result['claim']}")
                            st.caption(f"**ID:** {result['claim_id']}")
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                            st.caption(f"Time: {result.get('processing_time_ms', 0)}ms")
            else:
                st.info("No fact checks yet. Submit a claim to get started!")
        else:
            st.error("Unable to retrieve results")
    except Exception as e:
        st.error(f"Error loading recent checks: {str(e)}")

elif page == "Statistics":
    st.header("System Statistics")

    stats = get_statistics()

    if stats:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Claims Processed", stats.get('total_claims', 0))
        with col2:
            st.metric("Average Confidence", f"{stats.get('average_confidence', 0):.1%}")
        with col3:
            st.metric("Avg Processing Time", f"{stats.get('average_processing_time_ms', 0):.0f} ms")

        # Verdict distribution pie chart
        if stats.get('verdict_distribution'):
            st.subheader("Verdict Distribution")

            fig = px.pie(
                values=list(stats['verdict_distribution'].values()),
                names=list(stats['verdict_distribution'].keys()),
                title="Distribution of Verdicts"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No statistics available yet.")

elif page == "About":
    st.header("About This System")

    st.markdown("""
    ### Multi-Agent Misinformation Detection System

    This system uses a coordinated multi-agent approach to automatically detect and verify
    misinformation from various sources.

    #### Architecture

    The system employs **5 specialized agents**:

    1. **Classifier Agent**: Categorizes and decomposes claims
    2. **Collector Agent**: Gathers evidence from multiple sources
    3. **Anomaly Detector Agent**: Identifies suspicious patterns
    4. **Fact Checker Agent**: Verifies claims using RAG and reasoning
    5. **Reporter Agent**: Consolidates results and generates reports

    #### Technologies

    - **Framework**: LangGraph for multi-agent orchestration
    - **API**: FastAPI for REST endpoints
    - **Dashboard**: Streamlit for visualization
    - **NLP**: Sentence Transformers, spaCy
    - **LLMs**: Claude, GPT-4, Mistral (configurable)

    #### Features

    - Automated claim verification
    - Multi-source evidence collection
    - Semantic anomaly detection
    - Chain-of-thought reasoning
    - Credibility scoring
    - Detailed audit trails

    ---

    **Version**: 1.0.0
    **License**: MIT
    **Repository**: [GitHub](https://github.com/your-repo)
    """)

    st.info("💡 This is a research prototype. Always verify critical information with trusted sources.")
