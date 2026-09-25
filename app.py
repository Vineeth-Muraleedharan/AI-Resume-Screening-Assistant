"""
Streamlit UI for the AI Resume Screening Assistant.
Handles PDF upload, JD input, running evaluations, and displaying results
with Plotly gauge charts and skill comparison charts.
"""

import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from screener.ingest import load_and_split_multiple
from screener.store import (
    build_candidate_stores,
    get_candidates,
    retrieve_for_candidate,
    format_retrieved_context,
)
from screener.chain import evaluate_candidate
from screener.schema import ResumeEvaluation

MAX_FILES = 5
MAX_FILE_SIZE_MB = 5

st.set_page_config(page_title="AI Resume Screening Assistant", page_icon="📄", layout="wide")

NEON_GREEN = "#39FF14"
NEON_BLUE = "#00E5FF"
NEON_GOLD = "#FFD700"
NEON_RED = "#FF073A"
BG_BLACK = "#050505"

st.markdown(
    f"""
    <style>
    .stApp {{
    background:
        radial-gradient(circle at 15% 10%, rgba(57,255,20,0.09) 0%, rgba(0,0,0,0) 35%),
        radial-gradient(circle at 85% 85%, rgba(57,255,20,0.07) 0%, rgba(0,0,0,0) 40%),
        repeating-linear-gradient(0deg, rgba(57,255,20,0.02) 0px, rgba(57,255,20,0.02) 1px, transparent 1px, transparent 40px),
        repeating-linear-gradient(90deg, rgba(57,255,20,0.02) 0px, rgba(57,255,20,0.02) 1px, transparent 1px, transparent 40px),
        {BG_BLACK};
    background-attachment: fixed;
    border: 2px solid transparent;
    border-image: linear-gradient(135deg, {NEON_BLUE}, {NEON_GREEN}) 1;
    box-shadow: 0 0 18px rgba(0,229,255,0.35), 0 0 18px rgba(57,255,20,0.35) inset;
    box-sizing: border-box;
}}

    h1, h2, h3 {{
        color: {NEON_GREEN} !important;
        text-shadow: 0 0 8px rgba(57,255,20,0.45);
    }}

    p, span, label, .stMarkdown, .stCaption {{
        color: #e0e0e0;
    }}

    .stButton > button {{
        background-color: {NEON_GREEN};
        color: #000000;
        border: 1px solid {NEON_GREEN};
        border-radius: 8px;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(57,255,20,0.55);
        transition: all 0.15s ease-in-out;
    }}
    .stButton > button:hover {{
        box-shadow: 0 0 22px rgba(57,255,20,0.9);
        transform: translateY(-1px);
        color: #000000;
    }}

    .stTextArea textarea {{
        background-color: rgba(15,15,15,0.75) !important;
        border: 1px solid rgba(57,255,20,0.35) !important;
        border-radius: 8px !important;
        color: #e8ffe8 !important;
    }}

    [data-testid="stFileUploader"] {{
        background-color: rgba(15,15,15,0.55);
        border: 1px dashed rgba(57,255,20,0.4);
        border-radius: 10px;
        padding: 8px;
    }}

    .streamlit-expanderHeader {{
        background-color: rgba(15,15,15,0.65) !important;
        border: 1px solid rgba(57,255,20,0.25) !important;
        border-radius: 8px !important;
        color: #d8ffd8 !important;
    }}

    hr {{
        border-color: rgba(57,255,20,0.3) !important;
    }}

    [data-testid="stAlert"] {{
        background-color: rgba(15,15,15,0.8) !important;
        border-left: 4px solid {NEON_GREEN} !important;
    }}

    div[role="radiogroup"] label {{
        color: #e0e0e0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    if score >= 70:
        return NEON_GREEN
    elif score >= 40:
        return NEON_GOLD
    return NEON_RED


def recommendation_badge_html(recommendation: str) -> str:
    colors = {
        "Strong Fit": NEON_GREEN,
        "Possible Fit": NEON_GOLD,
        "Not a Fit": NEON_RED,
    }
    color = colors.get(recommendation, "#888888")
    return (
        f'<span style="background-color:{color};color:#000000;padding:4px 14px;'
        f'border-radius:12px;font-weight:700;font-size:0.9rem;'
        f'box-shadow:0 0 10px {color}99;">{recommendation}</span>'
    )


def make_gauge(score: int, title: str = "Match Score") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 16, "color": NEON_GREEN}},
            number={"font": {"color": NEON_GREEN, "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": NEON_GREEN, "tickfont": {"color": "#e0e0e0"}},
                "bar": {"color": score_color(score)},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "rgba(57,255,20,0.4)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,7,58,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,215,0,0.15)"},
                    {"range": [70, 100], "color": "rgba(57,255,20,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    return fig


def make_skills_donut(matching_count: int, missing_count: int) -> go.Figure:
    total = matching_count + missing_count
    coverage_pct = round((matching_count / total) * 100) if total > 0 else 0

    fig = go.Figure(
        go.Pie(
            labels=["Matching", "Missing"],
            values=[matching_count, missing_count],
            hole=0.65,
            marker=dict(colors=[NEON_GREEN, NEON_RED], line=dict(color=BG_BLACK, width=2)),
            textinfo="value",
            textfont={"color": "#000000", "size": 14},
            sort=False,
        )
    )
    fig.update_layout(
        annotations=[
            dict(text=f"{coverage_pct}%", x=0.5, y=0.5, font_size=28, font_color=NEON_GREEN, showarrow=False)
        ],
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font={"color": "#e0e0e0"}),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_ranking_bar(results: list[ResumeEvaluation]) -> go.Figure:
    ranked = sorted(results, key=lambda r: r.match_score, reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[r.match_score for r in ranked],
            y=[r.candidate for r in ranked],
            orientation="h",
            marker_color=[score_color(r.match_score) for r in ranked],
            text=[r.match_score for r in ranked],
            textposition="auto",
            textfont={"color": "#000000"},
        )
    )
    fig.update_layout(
        xaxis_title="Match Score",
        xaxis_range=[0, 100],
        height=100 + 60 * len(ranked),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
        xaxis_gridcolor="rgba(57,255,20,0.15)",
        yaxis_gridcolor="rgba(57,255,20,0.15)",
    )
    return fig


def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    dest_path = dest_dir / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def render_result_card(result: ResumeEvaluation):
    st.markdown(f"### {result.candidate}")
    st.markdown(recommendation_badge_html(result.recommendation), unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(make_gauge(result.match_score), use_container_width=True)
    with col2:
        st.plotly_chart(
            make_skills_donut(len(result.matching_skills), len(result.missing_skills)),
            use_container_width=True,
        )
        st.caption(f"{len(result.matching_skills)} matching · {len(result.missing_skills)} missing")

    if result.insufficient_jd_detail:
        st.warning("The job description was too vague to evaluate all criteria confidently.")

    st.write(result.summary)

    with st.expander("✅ Matching skills"):
        if result.matching_skills:
            for s in result.matching_skills:
                st.markdown(f"- {s}")
        else:
            st.write("None identified.")

    with st.expander("❌ Missing skills"):
        if result.missing_skills:
            for s in result.missing_skills:
                st.markdown(f"- {s}")
        else:
            st.write("None — all requirements covered.")

    with st.expander("💪 Strengths"):
        if result.strengths:
            for s in result.strengths:
                st.markdown(f"- {s}")
        else:
            st.write("None identified.")

    with st.expander("⚠️ Weaknesses"):
        if result.weaknesses:
            for s in result.weaknesses:
                st.markdown(f"- {s}")
        else:
            st.write("None identified.")

    with st.expander("📝 Justification"):
        st.write(result.justification)

    st.divider()


# ---------------------------------------------------------------------------
# Input section (main area — mobile-friendly, no sidebar drawer needed)
# ---------------------------------------------------------------------------

st.title("📄 AI Resume Screening Assistant")


input_col1, input_col2 = st.columns(2)

with input_col1:
    jd_text = st.text_area(
        "Job Description", height=220, placeholder="Paste the job description here..."
    )

with input_col2:
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )

mode = st.radio("Mode", ["Single / Compare", "Best Candidate"], horizontal=True)
run_clicked = st.button("Run Evaluation", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Main results area
# ---------------------------------------------------------------------------

if not run_clicked:
    st.info("Paste a job description, upload resume PDFs, and click **Run Evaluation** above.")
    st.stop()

# --- Validation ---
if not jd_text or not jd_text.strip():
    st.error("Please paste a job description before running the evaluation.")
    st.stop()

if not uploaded_files:
    st.error("Please upload at least one resume PDF.")
    st.stop()

if len(uploaded_files) > MAX_FILES:
    st.error(f"Please upload at most {MAX_FILES} resumes at a time.")
    st.stop()

for f in uploaded_files:
    size_mb = f.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"'{f.name}' is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit.")
        st.stop()

# --- Pipeline ---
with st.spinner("Reading resumes..."):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        saved_paths = [str(save_uploaded_file(f, tmp_path)) for f in uploaded_files]
        try:
            chunks = load_and_split_multiple(saved_paths)
        except ValueError as e:
            st.error(str(e))
            st.stop()

with st.spinner("Building resume index..."):
    try:
        stores = build_candidate_stores(chunks)
    except Exception as e:
        st.error(f"Failed to build resume index: {e}")
        st.stop()

candidates = get_candidates(stores)

results: list[ResumeEvaluation] = []
with st.spinner(f"Evaluating {len(candidates)} candidate(s)..."):
    for candidate in candidates:
        try:
            docs = retrieve_for_candidate(stores, candidate, jd_text, k=5)
            context = format_retrieved_context(docs)
            result = evaluate_candidate(candidate, jd_text, context)
            results.append(result)
        except Exception as e:
            st.error(f"Failed to evaluate '{candidate}': {e}")

if not results:
    st.error("No evaluations completed successfully.")
    st.stop()

# --- Display ---
if mode == "Single / Compare":
    if len(results) == 1:
        render_result_card(results[0])
    else:
        st.subheader("Score Comparison")
        st.plotly_chart(make_ranking_bar(results), use_container_width=True)
        st.divider()
        for result in sorted(results, key=lambda r: r.match_score, reverse=True):
            render_result_card(result)

else:  # Best Candidate
    best = max(results, key=lambda r: r.match_score)
    st.success(
        f"🏆 **Best Candidate: {best.candidate}** - Score {best.match_score}/100 ({best.recommendation})"
    )
    st.subheader("Ranking")
    st.plotly_chart(make_ranking_bar(results), use_container_width=True)
    st.divider()
    for result in sorted(results, key=lambda r: r.match_score, reverse=True):
        render_result_card(result)

if any(r.recommendation == "Strong Fit" for r in results):
    st.balloons()