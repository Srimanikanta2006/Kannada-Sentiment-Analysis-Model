import os
import sys
from io import BytesIO
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st


# Ensure project root is on sys.path so we can import agents.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agents.prediction_agent import PredictionAgent  # noqa: E402


st.set_page_config(
    page_title="Kannada Sentiment AI",
    layout="wide",
    page_icon="a",
)


CUSTOM_CSS = """
<style>
body {
    background: linear-gradient(135deg, #0F0C29 0%, #302B63 100%);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #ffffff;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    color: #111827;
}

.metric-card {
    text-align: center;
}

.metric-label {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
}

.metric-subtext {
    font-size: 0.9rem;
    color: #4B5563;
}

.sentiment-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    color: #ffffff;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
}

.sentiment-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}

.sentiment-subtitle {
    font-size: 0.95rem;
    opacity: 0.9;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(4px);
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #D1D5DB;
    margin-bottom: 0.25rem;
}

.section-subtitle {
    font-size: 0.92rem;
    color: #9CA3AF;
    margin-bottom: 0.75rem;
}

table.dataframe td, table.dataframe th {
    font-size: 0.9rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


if "prediction_agent" not in st.session_state:
    st.session_state["prediction_agent"] = None


def get_sentiment_color(sentiment: str) -> str:
    """
    Map sentiment labels to brand colors.
    """
    mapping: Dict[str, str] = {
        "happy": "#16A34A",   # green
        "sad": "#2563EB",     # blue
        "angry": "#DC2626",   # red
        "fear": "#EA580C",    # orange
        "disgust": "#7C3AED", # purple
        "neutral": "#6B7280", # gray
    }
    return mapping.get(sentiment.lower(), "#4B5563")


def sidebar() -> None:
    """
    Render sidebar with project information and model loading controls.
    """
    with st.sidebar:
        st.title("About this Project")
        st.markdown(
            """
**Model Type**  
Hybrid **MuRIL BERT** + **TF-IDF**  
Logistic Regression Ensemble

**Classes Supported**  
Happy · Sad · Angry · Fear · Disgust · Neutral

**Dataset**  
Custom Kannada social text with 6 sentiment classes.
"""
        )

        if st.button("Load Models", use_container_width=True):
            try:
                st.session_state["prediction_agent"] = PredictionAgent(model_dir="models")
                st.success("Models loaded successfully.")
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Error loading models: {exc}")


def header_section() -> None:
    """
    Render the main header and KPI metrics.
    """
    st.markdown(
        "<h1 style='text-align:center; color:#F9FAFB; font-weight:800;'>Kannada Sentiment Analysis </h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#E5E7EB; font-size:0.95rem;'>"
        "Hybrid MuRIL BERT + TF-IDF | 93%+ Accuracy"
        "</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card metric-card">
              <div class="metric-label">Accuracy</div>
              <div class="metric-value">93%</div>
              <div class="metric-subtext">On held-out Kannada validation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card metric-card">
              <div class="metric-label">Model</div>
              <div class="metric-value">MuRIL + TF-IDF</div>
              <div class="metric-subtext">Soft-voting logistic regression</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card metric-card">
              <div class="metric-label">Classes</div>
              <div class="metric-value">6</div>
              <div class="metric-subtext">Happy · Sad · Angry · Fear · Disgust · Neutral</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def single_prediction_section() -> None:
    """
    Render the single text prediction interface.
    """
    st.markdown("<div class='section-title'>Realtime Inference</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Paste any Kannada sentence and let the model "
        "decode its sentiment instantly.</div>",
        unsafe_allow_html=True,
    )

    with st.container():
        col_input, col_result = st.columns([2, 1])

        with col_input:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            input_text = st.text_area(
                "Enter Kannada text here ...",
                height=160,
                placeholder="Type or paste a Kannada sentence to analyze its sentiment...",
            )
            predict_clicked = st.button("Predict Sentiment", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            if predict_clicked:
                if not input_text or not input_text.strip():
                    st.error("Please enter some Kannada text before predicting.")
                elif st.session_state["prediction_agent"] is None:
                    st.error("Click **Load Models** in the sidebar to initialize the model.")
                else:
                    try:
                        agent: PredictionAgent = st.session_state["prediction_agent"]
                        results = agent.predict_texts([input_text])
                        if results:
                            result = results[0]
                            sentiment = str(result["sentiment"])
                            confidence = float(result["confidence"])
                            cleaned = str(result["cleaned"])
                            color = get_sentiment_color(sentiment)

                            st.markdown(
                                f"""
                                <div class="sentiment-card" style="background: linear-gradient(135deg, {color}, #111827);">
                                  <div class="badge">Predicted Sentiment</div>
                                  <div class="sentiment-title">{sentiment.title()}</div>
                                  <div class="sentiment-subtitle">
                                    Cleaned text: {cleaned[:120] + ("..." if len(cleaned) > 120 else "")}
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.write("Confidence")
                            st.progress(min(max(confidence / 100.0, 0.0), 1.0))
                        else:
                            st.warning("No prediction returned.")
                    except Exception as exc:  # pragma: no cover - UI feedback
                        st.error(f"Prediction failed: {exc}")


def batch_prediction_section() -> None:
    """
    Render the batch prediction interface for uploaded Excel files.
    """
    st.markdown("<div class='section-title'>Batch Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Upload an Excel file and score hundreds of "
        "Kannada sentences in one click.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Excel (.xlsx) with a 'kannada_text' column",
        type=["xlsx"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        if st.session_state["prediction_agent"] is None:
            st.error("Click **Load Models** in the sidebar to initialize the model.")
        else:
            try:
                df = pd.read_excel(uploaded_file)
                if "kannada_text" not in df.columns:
                    st.error("The uploaded Excel file must contain a 'kannada_text' column.")
                else:
                    texts = df["kannada_text"].astype(str).tolist()
                    agent: PredictionAgent = st.session_state["prediction_agent"]
                    results = agent.predict_texts(texts)
                    if not results:
                        st.warning("No predictions produced for the uploaded file.")
                    else:
                        results_df = pd.DataFrame(results)

                        def style_sentiment(val: str) -> str:
                            color = get_sentiment_color(str(val))
                            return f"background-color: {color}; color: #ffffff; font-weight: 600;"

                        styled = results_df.style.applymap(style_sentiment, subset=["sentiment"])
                        st.write("Preview of predictions:")
                        st.dataframe(styled, use_container_width=True)

                        buffer = BytesIO()
                        results_df.to_excel(buffer, index=False)
                        buffer.seek(0)

                        st.download_button(
                            "Download predictions.xlsx",
                            data=buffer,
                            file_name="predictions.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            ),
                            use_container_width=True,
                        )
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Batch prediction failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


def metrics_section() -> None:
    """
    Render model metrics, confusion matrix, and classification report.
    """
    st.markdown("<div class='section-title'>Model Diagnostics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Inspect how the ensemble behaves across "
        "all six sentiment classes.</div>",
        unsafe_allow_html=True,
    )

    cm_path = os.path.join(ROOT_DIR, "artifacts", "confusion_matrix.png")
    cr_path = os.path.join(ROOT_DIR, "artifacts", "classification_report.csv")

    cols = st.columns([1.4, 1])

    with cols[0]:
        if os.path.exists(cm_path):
            st.image(
                cm_path,
                caption="Hybrid Kannada Sentiment Analysis - Confusion Matrix",
                use_container_width=True,
            )
        else:
            st.info(
                "Confusion matrix not found yet. Train the model first with `python train.py`."
            )

    with cols[1]:
        if os.path.exists(cr_path):
            try:
                cr_df = pd.read_csv(cr_path)
                st.markdown("**Classification Report**")
                numeric_cols = ["precision", "recall", "f1-score", "support"]
                style = cr_df.style.background_gradient(
                    cmap="Blues",
                    subset=[c for c in numeric_cols if c in cr_df.columns],
                )
                st.dataframe(style, use_container_width=True)
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Failed to load classification report: {exc}")
        else:
            st.info(
                "Classification report not found yet. It will be generated after training."
            )


def main() -> None:
    """
    Render the full Streamlit dashboard for Kannada Sentiment Analysis.
    """
    sidebar()

    header_section()
    st.markdown("---")

    single_prediction_section()
    st.markdown("---")

    batch_prediction_section()
    st.markdown("---")

    metrics_section()


if __name__ == "__main__":
    main()

