from pathlib import Path
import streamlit as st


def load_styles():
    """Load and inject custom CSS into the Streamlit app."""
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def welcome_state():
    """Render the centered welcome state when chat is empty."""
    st.markdown("""
<div class="welcome-state">
    <div class="welcome-icon">⚖️</div>
    <div class="welcome-title">Ask me anything about compliance</div>
    <div class="welcome-subtitle">
        I can help you navigate HIPAA, GDPR, FINRA, CCPA, and the EU AI Act.<br>
        Select a topic from the sidebar or ask your question below.
    </div>
</div>
""", unsafe_allow_html=True)