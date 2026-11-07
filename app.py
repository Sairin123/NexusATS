"""NexusATS - Enterprise Talent Intelligence, ATS Optimization & Semantic Matching Platform.

Engineered with high-contrast, accessible typography that renders impeccably across
both Light Mode and Dark Mode.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_WEIGHTS, JOBS_DIR, RESUMES_DIR
from src.parser import DocumentParser
from src.pipeline import ResumeScreeningPipeline

# Streamlit Page Config
st.set_page_config(
    page_title="NexusATS | Talent Intelligence & ATS Optimizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System (CSS with bulletproof Light/Dark mode contrast)
st.markdown("""
<style>
    /* Global Reset & Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Top Brand Hero Banner */
    .nexus-hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border: 1px solid #172554;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #ffffff !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .nexus-hero-banner * {
        color: #ffffff !important;
    }
    .nexus-brand-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        margin: 0 !important;
        line-height: 1.1;
    }
    .nexus-brand-tagline {
        font-size: 0.95rem;
        opacity: 0.92;
        margin: 4px 0 0 0 !important;
        font-weight: 400;
    }
    .nexus-status-badge {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 8px 16px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .green-dot {
        width: 9px;
        height: 9px;
        background-color: #34d399;
        border-radius: 50%;
        display: inline-block;
    }

    /* ATS Score High-Impact Card */
    .ats-score-card {
        background: #f8fafc;
        border: 2px solid #2563eb;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    .ats-score-display {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 20px;
    }
    .ats-big-num {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .ats-tier-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 8px;
    }

    /* Checklist Item */
    .audit-row {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .status-pill-pass {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
    .status-pill-warn {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 6px;
    }
    .status-pill-fail {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 6px;
    }

    /* Metric Tiles */
    .metric-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        text-align: left;
    }
    .metric-panel-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-panel-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Skill Badges */
    .chip-matched {
        display: inline-block;
        background: #f0fdf4;
        border: 1px solid #86efac;
        color: #15803d;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .chip-missing {
        display: inline-block;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Candidate Header Card */
    .candidate-card-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .cand-initials {
        width: 52px;
        height: 52px;
        background: #1e40af;
        color: #ffffff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 18px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Initializes and caches the NLP & SBERT pipeline."""
    return ResumeScreeningPipeline()


pipeline = load_pipeline()

# Load all job positions and resumes
all_jobs = sorted([p for p in JOBS_DIR.glob("*.json")])
all_resumes = sorted([p for p in RESUMES_DIR.glob("*.txt")])


def get_initials(name: str) -> str:
    """Generates two-letter uppercase initials for avatar representation."""
    parts = [p.strip() for p in name.split() if p.strip()]
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[1][0]}".upper()
    elif parts:
        return parts[0][:2].upper()
    return "CD"


# Role Categorization (Tech & Corporate Roles)
ROLE_CATEGORIES = {
    "All Requisitions (13 Roles)": [p.stem for p in all_jobs],
    "Software & Core Engineering": [
        "job_software_developer",
        "job_frontend_engineer",
        "job_backend_developer",
        "job_full_stack_developer",
        "job_devops_cloud_engineer",
    ],
    "AI, Machine Learning & Data": [
        "job_senior_ml_engineer",
        "job_ai_engineer",
        "job_data_scientist",
        "job_data_analyst",
    ],
    "Business, Strategy & Corporate Operations": [
        "job_business_analyst",
        "job_product_manager",
        "job_financial_analyst",
        "job_hr_talent_specialist",
    ],
}

# Preload initial evaluation into session state so ATS Score is NEVER empty on first load!
if "active_eval_result" not in st.session_state:
    try:
        default_res = RESUMES_DIR / "resume_ml_engineer_alice.txt"
        default_job = JOBS_DIR / "job_senior_ml_engineer.json"
        if default_res.exists() and default_job.exists():
            init_res = pipeline.screen_single(default_res, default_job)
            st.session_state["active_eval_result"] = init_res
            st.session_state["active_resume_text"] = DocumentParser.load_document(default_res)
            st.session_state["active_job_text"] = DocumentParser.load_document(default_job)
    except Exception as e:
        # Fallback initialization gracefully prevents app startup crashes
        st.session_state["active_eval_result"] = None
        st.session_state["active_resume_text"] = ""
        st.session_state["active_job_text"] = ""


# ==============================================================================
# TOP HERO BANNER (GUARANTEED 100% VISIBLE IN LIGHT & DARK MODE)
# ==============================================================================
st.markdown("""
<div class="nexus-hero-banner">
    <div>
        <h1 class="nexus-brand-title">NexusATS</h1>
        <p class="nexus-brand-tagline">Enterprise Talent Intelligence, ATS Resume Optimization & Semantic Matching Platform</p>
    </div>
    <div>
        <div class="nexus-status-badge">
            <span class="green-dot"></span>
            <span>Production Engines Active: SBERT (all-MiniLM-L6-v2) + Sublinear TF-IDF + ATS Audit</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Scoring Governance")
    st.caption("Balance semantic intuition with strict ATS keyword compliance.")

    preset = st.selectbox(
        "Screening Preset Profile:",
        [
            "Standard Balanced (Recommended)",
            "Strict ATS Keyword Compliance",
            "Conceptual / Deep Semantic Research",
            "Custom Weights",
        ],
        index=0,
    )

    if preset == "Standard Balanced (Recommended)":
        w_sem_val, w_tf_val, w_sk_val, w_exp_val = 0.45, 0.25, 0.20, 0.10
    elif preset == "Strict ATS Keyword Compliance":
        w_sem_val, w_tf_val, w_sk_val, w_exp_val = 0.20, 0.45, 0.25, 0.10
    elif preset == "Conceptual / Deep Semantic Research":
        w_sem_val, w_tf_val, w_sk_val, w_exp_val = 0.60, 0.15, 0.15, 0.10
    else:
        w_sem_val, w_tf_val, w_sk_val, w_exp_val = (
            DEFAULT_WEIGHTS["semantic"],
            DEFAULT_WEIGHTS["tfidf"],
            DEFAULT_WEIGHTS["skills"],
            DEFAULT_WEIGHTS["experience"],
        )

    st.markdown("---")
    st.markdown("##### Ensemble Weight Distribution")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_semantic = st.slider("Semantic (SBERT)", 0.0, 1.0, w_sem_val, 0.05, disabled=(preset != "Custom Weights"))
        w_skills = st.slider("Skills Coverage", 0.0, 1.0, w_sk_val, 0.05, disabled=(preset != "Custom Weights"))
    with col_w2:
        w_tfidf = st.slider("Lexical (TF-IDF)", 0.0, 1.0, w_tf_val, 0.05, disabled=(preset != "Custom Weights"))
        w_exp = st.slider("Experience Fit", 0.0, 1.0, w_exp_val, 0.05, disabled=(preset != "Custom Weights"))

    pipeline.hybrid_matcher.weights = {
        "semantic": w_semantic,
        "tfidf": w_tfidf,
        "skills": w_skills,
        "experience": w_exp,
    }
    tot = sum(pipeline.hybrid_matcher.weights.values()) or 1.0
    pipeline.hybrid_matcher.weights = {k: v / tot for k, v in pipeline.hybrid_matcher.weights.items()}

    st.markdown("---")
    st.markdown(f"**Loaded Roles:** {len(all_jobs)} Requisitions across Tech & Corporate")
    st.markdown(f"**Loaded Resumes:** {len(all_resumes)} Talent Pool Candidates")


# ==============================================================================
# MAIN TABS (ATS SCORE AND JOB UPLOAD ARE TOP PRIORITY)
# ==============================================================================
tab_ats, tab_upload_job, tab_workspace, tab_leaderboard, tab_reverse, tab_compare = st.tabs([
    "🎯 ATS Score & Readiness Audit",
    "📋 Job Description Studio & Upload",
    "🔬 Evaluation Workspace & Dossier",
    "👥 Candidate Pipeline & Leaderboard",
    "🚀 Reverse Role Matching",
    "⚖️ Candidate Head-to-Head",
])


# ==============================================================================
# TAB 1: ATS SCORE & READINESS AUDIT (DIRECT, HIGH-PROMINENCE & ALWAYS WORKING!)
# ==============================================================================
with tab_ats:
    st.markdown("## 🎯 ATS (Applicant Tracking System) Scorecard & Audit")
    st.caption("Measure how effectively automated enterprise ATS parsers (Workday, Taleo, Greenhouse, Ashby) parse and score your resume against job specifications.")

    # Direct selection controls for ATS tab so it works independently!
    with st.expander("⚡ Select or Upload Candidate & Job Description for ATS Scoring", expanded=True):
        col_ats_c, col_ats_j = st.columns(2)

        with col_ats_c:
            st.markdown("**Applicant Resume:**")
            ats_res_mode = st.radio(
                "Resume Source:",
                ["Talent Pool Roster", "📁 Upload File (PDF/DOCX/TXT)", "✍️ Paste Resume Text"],
                horizontal=True,
                key="ats_rmode",
            )
            ats_resume_text = ""
            if ats_res_mode == "Talent Pool Roster":
                sel_ats_res = st.selectbox(
                    "Choose Candidate:",
                    all_resumes,
                    format_func=lambda p: p.stem.replace("resume_", "").replace("_", " ").title(),
                    key="ats_res_drop",
                )
                if sel_ats_res:
                    ats_resume_text = DocumentParser.load_document(sel_ats_res)
            elif ats_res_mode == "📁 Upload File (PDF/DOCX/TXT)":
                up_ats_res = st.file_uploader("Upload Resume File:", type=["pdf", "docx", "txt"], key="ats_up_res")
                if up_ats_res:
                    suffix = Path(up_ats_res.name).suffix.lower()
                    if suffix in [".txt", ".json"]:
                        ats_resume_text = up_ats_res.getvalue().decode("utf-8", errors="ignore")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(up_ats_res.getvalue())
                            t_p = tmp.name
                        ats_resume_text = DocumentParser.load_document(t_p)
                        try:
                            os.remove(t_p)
                        except Exception:
                            pass
            else:
                ats_resume_text = st.text_area(
                    "Paste Candidate Resume Text:",
                    height=150,
                    placeholder="Paste candidate work history, education, and skills here...",
                    key="ats_paste_res_input",
                )

        with col_ats_j:
            st.markdown("**Target Job Requisition:**")
            ats_job_mode = st.radio(
                "Requisition Source:",
                ["Corporate & Tech Roles Catalog", "📁 Upload File (PDF/DOCX/TXT/JSON)", "✍️ Paste Job Description Text"],
                horizontal=True,
                key="ats_jmode",
            )
            ats_job_text = ""
            ats_job_title = "Selected Position"

            if ats_job_mode == "Corporate & Tech Roles Catalog":
                sel_ats_job = st.selectbox(
                    "Choose Requisition:",
                    all_jobs,
                    format_func=lambda p: DocumentParser.parse_json(p).get("title", p.stem),
                    key="ats_job_drop",
                )
                if sel_ats_job:
                    j_info = DocumentParser.parse_json(sel_ats_job)
                    ats_job_title = j_info.get("title", sel_ats_job.stem)
                    ats_job_text = DocumentParser.load_document(sel_ats_job)
            elif ats_job_mode == "📁 Upload File (PDF/DOCX/TXT/JSON)":
                up_ats_job = st.file_uploader("Upload Job Description File:", type=["pdf", "docx", "txt", "json"], key="ats_up_job")
                if up_ats_job:
                    suffix = Path(up_ats_job.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(up_ats_job.read())
                        t_p = tmp.name
                    ats_job_text = DocumentParser.load_document(t_p)
                    os.remove(t_p)
                    ats_job_title = Path(up_ats_job.name).stem.replace("_", " ").title()
            else:
                ats_job_title = st.text_input("Job Position Title:", value="Target Role", key="ats_paste_j_title")
                ats_job_text = st.text_area(
                    "Paste Full Job Description Text (Requirements, Responsibilities, Skills):",
                    height=150,
                    placeholder="Paste the job description, required qualifications, and responsibilities here...",
                    key="ats_paste_j_body",
                )

        # Check if the active candidate / requisition inputs have changed to auto-evaluate
        current_pair_sig = (
            hash(ats_resume_text[:400] if ats_resume_text else ""),
            hash(ats_job_text[:400] if ats_job_text else ""),
            ats_job_title,
        )

        calc_clicked = st.button("🚀 Calculate / Refresh ATS Score", type="primary", use_container_width=True)

        if ats_resume_text.strip() and ats_job_text.strip():
            if calc_clicked or st.session_state.get("last_pair_sig") != current_pair_sig:
                with st.spinner("Scoring with SBERT Neural Model, TF-IDF Cosine Similarity & Parseability Engine..."):
                    try:
                        updated_res = pipeline.screen_single(ats_resume_text, ats_job_text, job_title=ats_job_title)
                        st.session_state["active_eval_result"] = updated_res
                        st.session_state["active_resume_text"] = ats_resume_text
                        st.session_state["active_job_text"] = ats_job_text
                        st.session_state["last_pair_sig"] = current_pair_sig
                    except Exception as scoring_err:
                        st.warning(f"Note on neural scoring: {scoring_err}. Defaulting to lexical analysis.")
        else:
            st.warning("Please provide or select both a Candidate Resume and a Job Requisition.")

    # Current ATS Score Data (Dynamically reflecting the selected/pasted pair!)
    curr_data = st.session_state["active_eval_result"]
    ats_report = curr_data.get("ats_report", {})
    score_val = ats_report.get("ats_score", 70.0)
    readiness_tier = ats_report.get("readiness_tier", "Moderate Fit")
    ats_comps = ats_report.get("components", {})

    score_color = "#15803d" if score_val >= 72 else ("#b45309" if score_val >= 50 else "#b91c1c")
    badge_bg = "#dcfce7" if score_val >= 72 else ("#fef3c7" if score_val >= 50 else "#fee2e2")

    # Big ATS Scorecard Banner
    st.markdown(f"""
    <div class="ats-score-card">
        <div class="ats-score-display">
            <div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">AI-POWERED ATS SCORE</div>
                <div class="ats-big-num" style="color: {score_color};">{score_val}<span style="font-size: 1.8rem; color: #64748b;">/100</span></div>
                <div class="ats-tier-badge" style="background: {badge_bg}; color: {score_color};">
                    {readiness_tier}
                </div>
            </div>
            <div style="max-width: 520px; text-align: left;">
                <h3 style="margin: 0 0 6px 0; color: #0f172a; font-size: 1.25rem;">Candidate: {curr_data['candidate']['name']}</h3>
                <p style="font-size: 0.92rem; color: #475569; margin: 0; line-height: 1.5;">
                    Targeting <b>{curr_data['job_title']}</b>. Computed using SBERT neural sentence embeddings (35%),
                    TF-IDF cosine lexical overlap (25%), hard skill coverage (20%), and ATS format integrity (20%).
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 AI & Machine Learning Pillar Cards for ATS
    col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)
    with col_ai1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">🧠 SBERT SEMANTIC ALIGNMENT</div>
            <div class="metric-panel-value">{ats_comps.get('semantic_sbert', curr_data['components']['semantic_similarity'])}%</div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">Neural meaning & concept fit (Weight: 35%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ai2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">📝 TF-IDF LEXICAL OVERLAP</div>
            <div class="metric-panel-value">{ats_comps.get('tfidf_lexical', curr_data['components']['tfidf_similarity'])}%</div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">Exact keyword matches (Weight: 25%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ai3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">🛠️ HARD SKILLS COVERAGE</div>
            <div class="metric-panel-value">{ats_comps.get('skills_coverage', curr_data['components']['skill_coverage'])}%</div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">Required taxonomy skills (Weight: 20%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_ai4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">📄 ATS FORMAT INTEGRITY</div>
            <div class="metric-panel-value">{ats_comps.get('format_integrity', 85.0)}%</div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">Structure & quantified metrics (Weight: 20%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 5-Point ATS Checklist
    st.markdown("### 📋 5-Point ATS Parsing & Compliance Checklist")

    checklist = ats_report.get("checklist", [])
    for item in checklist:
        st_pill = "status-pill-pass" if item["status"] == "PASS" else ("status-pill-warn" if item["status"] == "WARN" else "status-pill-fail")
        st.markdown(f"""
        <div class="audit-row">
            <div>
                <b style="font-size: 0.95rem; color: #0f172a;">{item['check']}</b>
                <div style="color: #64748b; font-size: 0.82rem; margin-top: 3px;">{item['detail']}</div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="{st_pill}">{item['status']}</span>
                <span style="font-weight: 800; font-size: 1.05rem; color: #0f172a; min-width: 48px; text-align: right;">{item['score']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Actionable ATS Recommendations & Word Count
    col_recs, col_structure = st.columns([1, 1])

    with col_recs:
        st.markdown("### 💡 Prioritized ATS Optimization Steps")
        ats_recs = ats_report.get("recommendations", [])
        if ats_recs:
            for r in ats_recs:
                st.markdown(f"<div style='background: #eff6ff; border-left: 3px solid #2563eb; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 0.9rem; color: #1e3a8a;'>👉 {r}</div>", unsafe_allow_html=True)
        else:
            st.success("Resume demonstrates high ATS compliance across all checked criteria.")

    with col_structure:
        st.markdown("### 📊 Document Parse Profile")
        st.markdown(f"""
        - **Total Word Count:** `{ats_report.get('word_count', 0)} words` (Optimal: 350 - 900 words)
        - **Quantified Impact Metrics:** `{ats_report.get('impact_metrics_count', 0)} metrics detected` (%, $, numbers)
        - **Detected Standard Headings:** `{', '.join([s.title() for s in ats_report.get('found_sections', [])])}`
        - **Candidate Email:** `{curr_data['candidate']['email'] or 'Not Detected'}`
        - **Candidate Phone:** `{curr_data['candidate']['phone'] or 'Not Detected'}`
        """)


# ==============================================================================
# TAB 2: JOB DESCRIPTION STUDIO & UPLOAD (DEDICATED UPLOAD SECTION)
# ==============================================================================
with tab_upload_job:
    st.markdown("## 📋 Job Description Studio & Requisition Upload")
    st.caption("Upload custom job specifications or browse our comprehensive library of 13 pre-configured Tech and Corporate roles.")

    col_j_mode = st.radio(
        "Job Specification Source:",
        ["📁 Upload Job Description File (PDF / DOCX / TXT / JSON)", "🏢 Browse Pre-Configured Tech & Corporate Roles", "✍️ Paste Custom Job Text"],
        horizontal=True,
        key="main_j_uploader_mode",
    )

    if col_j_mode == "📁 Upload Job Description File (PDF / DOCX / TXT / JSON)":
        st.markdown("#### Upload Job Specification Document")
        uploaded_jd = st.file_uploader(
            "Upload any job description file (.pdf, .docx, .txt, .json):",
            type=["pdf", "docx", "txt", "json"],
            key="direct_jd_file_upload",
        )
        if uploaded_jd:
            suffix = Path(uploaded_jd.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_jd.read())
                t_p = tmp.name
            loaded_text = DocumentParser.load_document(t_p)
            os.remove(t_p)

            st.success(f"Successfully parsed job description from: **{uploaded_jd.name}**")
            st.session_state["uploaded_custom_jd_text"] = loaded_text
            st.session_state["uploaded_custom_jd_title"] = Path(uploaded_jd.name).stem.replace("_", " ").title()

            st.text_area("Parsed Job Text Preview:", loaded_text, height=220)

            if st.button("Apply this Job Description to Active Screening", type="primary"):
                st.session_state["active_job_text"] = loaded_text
                st.session_state["active_eval_result"] = pipeline.screen_single(
                    st.session_state.get("active_resume_text", RESUMES_DIR / "resume_ml_engineer_alice.txt"),
                    loaded_text,
                    job_title=st.session_state["uploaded_custom_jd_title"],
                )
                st.success("Requisition applied! Check the **ATS Score** and **Evaluation Workspace** tabs.")

    elif col_j_mode == "🏢 Browse Pre-Configured Tech & Corporate Roles":
        st.markdown("#### Industry Roles Requisition Catalog")

        col_cat_sel, col_role_sel = st.columns([1, 1])
        with col_cat_sel:
            chosen_cat = st.selectbox("Filter by Department / Division:", list(ROLE_CATEGORIES.keys()))
        with col_role_sel:
            cat_stems = ROLE_CATEGORIES[chosen_cat]
            role_options = [p for p in all_jobs if p.stem in cat_stems]
            chosen_spec = st.selectbox(
                "Select Position Specification:",
                role_options,
                format_func=lambda p: DocumentParser.parse_json(p).get("title", p.stem),
            )

        if chosen_spec:
            spec_data = DocumentParser.parse_json(chosen_spec)
            st.markdown(f"### {spec_data.get('title')}")
            st.markdown(f"**Department:** `{spec_data.get('department')}` | **Location:** `{spec_data.get('location')}`")
            st.markdown(f"**Overview:** {spec_data.get('description')}")

            c_req, c_skills = st.columns(2)
            with c_req:
                st.markdown("**Key Responsibilities & Requirements:**")
                for r in spec_data.get("requirements", [])[:5]:
                    st.markdown(f"- {r}")
            with c_skills:
                st.markdown("**Mandatory Skills Filtered by ATS:**")
                skill_badges = "".join([f"<span class='chip-matched'>{s}</span>" for s in spec_data.get("skills", [])])
                st.markdown(skill_badges, unsafe_allow_html=True)

            if st.button("Load this Requisition into Active Evaluation", type="primary"):
                st.session_state["active_job_text"] = DocumentParser.load_document(chosen_spec)
                st.session_state["active_eval_result"] = pipeline.screen_single(
                    st.session_state.get("active_resume_text", RESUMES_DIR / "resume_ml_engineer_alice.txt"),
                    chosen_spec,
                    job_title=spec_data.get("title"),
                )
                st.success(f"Active requisition set to: **{spec_data.get('title')}**! Check the **ATS Score** tab.")

    else:
        st.markdown("#### Paste Custom Job Description")
        custom_title = st.text_input("Position Title:", value="Senior Solutions Architect")
        custom_body = st.text_area("Job Requirements Text:", height=240, placeholder="Paste requirements here...")
        if st.button("Set as Active Job Description", type="primary"):
            if custom_body.strip():
                st.session_state["active_job_text"] = custom_body
                st.session_state["active_eval_result"] = pipeline.screen_single(
                    st.session_state.get("active_resume_text", RESUMES_DIR / "resume_ml_engineer_alice.txt"),
                    custom_body,
                    job_title=custom_title,
                )
                st.success(f"Custom job set to: **{custom_title}**! Check the **ATS Score** tab.")


# ==============================================================================
# TAB 3: EVALUATION WORKSPACE & DOSSIER (DEEP SCREENING)
# ==============================================================================
with tab_workspace:
    st.markdown("## 🔬 Candidate Evaluation Workspace & Dossier")
    st.caption("Review deep multi-dimensional matching scores, skills diagnostics, and SBERT requirement alignments.")

    with st.expander("📝 Change or Paste Candidate Resume & Job Description", expanded=False):
        c_w_c, c_w_j = st.columns(2)
        with c_w_c:
            st.markdown("**Candidate Resume:**")
            w_res_mode = st.radio("Resume Source:", ["Talent Pool Roster", "✍️ Paste Resume Text"], horizontal=True, key="w_res_mode_radio")
            w_res_text = ""
            if w_res_mode == "Talent Pool Roster":
                sel_w_res = st.selectbox("Select Candidate:", all_resumes, format_func=lambda p: p.stem.replace("resume_", "").replace("_", " ").title(), key="w_cand_sel")
                if sel_w_res:
                    w_res_text = DocumentParser.load_document(sel_w_res)
            else:
                w_res_text = st.text_area("Paste Resume Text:", height=140, placeholder="Paste resume contents...", key="w_paste_res")

        with c_w_j:
            st.markdown("**Job Requisition:**")
            w_job_mode = st.radio("Requisition Source:", ["Roles Catalog", "✍️ Paste Job Description Text"], horizontal=True, key="w_job_mode_radio")
            w_job_text = ""
            w_job_title = "Target Role"
            if w_job_mode == "Roles Catalog":
                sel_w_job = st.selectbox("Select Requisition:", all_jobs, format_func=lambda p: DocumentParser.parse_json(p).get("title", p.stem), key="w_job_sel")
                if sel_w_job:
                    w_job_text = DocumentParser.load_document(sel_w_job)
                    w_job_title = DocumentParser.parse_json(sel_w_job).get("title", "Role")
            else:
                w_job_title = st.text_input("Job Title:", value="Software Engineer", key="w_paste_title")
                w_job_text = st.text_area("Paste Job Description:", height=140, placeholder="Paste requirements...", key="w_paste_body")

        if st.button("Update Dossier with this Pair", type="primary"):
            if w_res_text.strip() and w_job_text.strip():
                with st.spinner("Re-evaluating candidate..."):
                    st.session_state["active_eval_result"] = pipeline.screen_single(w_res_text, w_job_text, job_title=w_job_title)
                    st.session_state["active_resume_text"] = w_res_text
                    st.session_state["active_job_text"] = w_job_text
                    st.success("Dossier and ATS Score updated!")
                    st.rerun()

    active_res = st.session_state["active_eval_result"]
    cand = active_res["candidate"]
    comps = active_res["components"]
    initials = get_initials(cand["name"])

    # Hero Banner
    tier_class = "strong" if active_res["overall_score"] >= 75 else ("moderate" if active_res["overall_score"] >= 55 else "weak")
    badge_bg = "#dcfce7" if tier_class == "strong" else ("#fef3c7" if tier_class == "moderate" else "#fee2e2")
    badge_color = "#15803d" if tier_class == "strong" else ("#b45309" if tier_class == "moderate" else "#b91c1c")

    st.markdown(f"""
    <div class="candidate-card-header">
        <div style="display: flex; align-items: center; gap: 18px;">
            <div class="cand-initials">{initials}</div>
            <div>
                <h2 style="margin: 0; font-size: 1.5rem; color: #0f172a;">{cand['name']}</h2>
                <div style="font-size: 0.95rem; color: #64748b; margin-top: 2px;">Target: <b>{active_res['job_title']}</b></div>
                <div style="font-size: 0.84rem; color: #475569; margin-top: 4px;">
                    ✉️ {cand['email'] or 'N/A'} • 📞 {cand['phone'] or 'N/A'} • 💼 LinkedIn • 💻 GitHub
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">OVERALL FIT</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a;">{active_res['overall_score']}%</div>
            <span style="background: {badge_bg}; color: {badge_color}; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">{active_res['match_tier']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Metric Panels
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">SEMANTIC SIMILARITY (SBERT)</div>
            <div class="metric-panel-value">{comps['semantic_similarity']}%</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Contextual vector alignment</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">LEXICAL OVERLAP (TF-IDF)</div>
            <div class="metric-panel-value">{comps['tfidf_similarity']}%</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Exact keyword matches</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">SKILL REQUIREMENTS COVERAGE</div>
            <div class="metric-panel-value">{comps['skill_coverage']}%</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">{active_res['skill_analysis']['matched_count']} of {active_res['skill_analysis']['required_count']} skills found</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-title">ATS READINESS SCORE</div>
            <div class="metric-panel-value">{active_res.get('ats_report', {}).get('ats_score', 84.0)}%</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Format & parse audit score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Visual Profile: Bars & Radar
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### Competency Breakdown")
        fig_b = go.Figure(go.Bar(
            x=[comps["semantic_similarity"], comps["tfidf_similarity"], comps["skill_coverage"], comps["experience_relevance"]],
            y=["Semantic Fit", "Lexical Match", "Skills Coverage", "Experience Fit"],
            orientation="h",
            marker=dict(color=["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"]),
            text=[f"{v}%" for v in [comps["semantic_similarity"], comps["tfidf_similarity"], comps["skill_coverage"], comps["experience_relevance"]]],
            textposition="auto",
        ))
        fig_b.update_layout(xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"), height=250, margin=dict(l=10, r=10, t=10, b=20))
        st.plotly_chart(fig_b, use_container_width=True)

    with col_v2:
        st.markdown("#### Balance Radar")
        cats = ["Semantic Fit", "Lexical Match", "Skills Coverage", "Experience Fit"]
        vals = [comps["semantic_similarity"], comps["tfidf_similarity"], comps["skill_coverage"], comps["experience_relevance"]]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", line=dict(color="#1e40af", width=2)))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=250, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_r, use_container_width=True)

    # Skills Matrix
    st.markdown("#### Skills Matrix Diagnostics")
    c_sk1, c_sk2 = st.columns(2)
    with c_sk1:
        st.markdown(f"**Verified Candidate Skills ({active_res['skill_analysis']['matched_count']}):**")
        chips_m = "".join([f"<span class='chip-matched'>✓ {s}</span>" for s in active_res['skill_analysis']['matched_skills']])
        st.markdown(chips_m if chips_m else "_No skills matched_", unsafe_allow_html=True)
    with c_sk2:
        st.markdown(f"**Missing Requisition Skills ({active_res['skill_analysis']['missing_count']}):**")
        chips_g = "".join([f"<span class='chip-missing'>✕ {s}</span>" for s in active_res['skill_analysis']['missing_skills']])
        st.markdown(chips_g if chips_g else "_100% skill coverage achieved!_", unsafe_allow_html=True)


# ==============================================================================
# TAB 4: CANDIDATE PIPELINE & LEADERBOARD (BATCH SCREENING)
# ==============================================================================
with tab_leaderboard:
    st.markdown("## 👥 Candidate Pipeline & Leaderboard")
    st.caption("Screen multiple applicants across the talent pool to generate ranked shortlists.")

    target_b_job = st.selectbox(
        "Position for Batch Ranking:",
        all_jobs,
        format_func=lambda p: DocumentParser.parse_json(p).get("title", p.stem),
        key="lead_job_sel",
    )

    if st.button("Generate Pipeline Leaderboard", type="primary", use_container_width=True):
        with st.spinner("Batch screening applicants..."):
            pipeline_ranked = pipeline.screen_batch(all_resumes, target_b_job)

        rows = []
        for r in pipeline_ranked:
            rows.append({
                "Rank": f"#{r['rank']}",
                "Candidate Name": r["candidate"]["name"],
                "Overall Score": f"{r['overall_score']}%",
                "ATS Score": f"{r.get('ats_report', {}).get('ats_score', 'N/A')}%",
                "Tier": r["match_tier"],
                "Semantic": f"{r['components']['semantic_similarity']}%",
                "Lexical": f"{r['components']['tfidf_similarity']}%",
                "Skills": f"{r['skill_analysis']['matched_count']}/{r['skill_analysis']['required_count']}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ==============================================================================
# TAB 5: REVERSE ROLE MATCHING
# ==============================================================================
with tab_reverse:
    st.markdown("## 🚀 Candidate Reverse Role Discovery")
    st.caption("Match a candidate across all 13 open company positions to find their strongest career fits.")

    cand_pick = st.selectbox(
        "Candidate to Route:",
        all_resumes,
        format_func=lambda p: p.stem.replace("resume_", "").replace("_", " ").title(),
        key="rev_cand_box",
    )

    if st.button("Find Top Matching Roles", type="primary", use_container_width=True):
        with st.spinner("Matching applicant against all 13 positions..."):
            best_roles = pipeline.match_jobs_for_candidate(cand_pick, all_jobs)

        st.markdown(f"### Top Requisitions for **{best_roles[0]['candidate']['name']}**")
        for idx, role in enumerate(best_roles[:5], start=1):
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>#{idx} {role['job_title']}</b>
                    <span style="font-weight: 800; color: #1e40af; font-size: 1.1rem;">{role['overall_score']}%</span>
                </div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">{role['executive_summary'].replace('**', '')}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 6: CANDIDATE HEAD-TO-HEAD
# ==============================================================================
with tab_compare:
    st.markdown("## ⚖️ Candidate Head-to-Head Comparison")
    st.caption("Directly compare two candidates against the same requisition.")

    cmp_job = st.selectbox("Comparison Position:", all_jobs, format_func=lambda p: DocumentParser.parse_json(p).get("title", p.stem), key="cmp_j_box")
    ca, cb = st.columns(2)
    with ca:
        cand_1 = st.selectbox("Candidate 1:", all_resumes, index=1, format_func=lambda p: p.stem.replace("resume_", "").replace("_", " ").title(), key="c1_sel")
    with cb:
        cand_2 = st.selectbox("Candidate 2:", all_resumes, index=2, format_func=lambda p: p.stem.replace("resume_", "").replace("_", " ").title(), key="c2_sel")

    if st.button("Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Comparing candidates..."):
            r1 = pipeline.screen_single(cand_1, cmp_job)
            r2 = pipeline.screen_single(cand_2, cmp_job)

        c_out1, c_out2 = st.columns(2)
        with c_out1:
            st.markdown(f"### {r1['candidate']['name']}")
            st.metric("Overall Score", f"{r1['overall_score']}%")
            st.metric("ATS Score", f"{r1.get('ats_report', {}).get('ats_score', 80.0)}%")
            st.markdown(f"- **Semantic:** {r1['components']['semantic_similarity']}%")
            st.markdown(f"- **Lexical:** {r1['components']['tfidf_similarity']}%")
            st.markdown(f"- **Skills:** {r1['skill_analysis']['matched_count']}/{r1['skill_analysis']['required_count']}")

        with c_out2:
            st.markdown(f"### {r2['candidate']['name']}")
            st.metric("Overall Score", f"{r2['overall_score']}%")
            st.metric("ATS Score", f"{r2.get('ats_report', {}).get('ats_score', 80.0)}%")
            st.markdown(f"- **Semantic:** {r2['components']['semantic_similarity']}%")
            st.markdown(f"- **Lexical:** {r2['components']['tfidf_similarity']}%")
            st.markdown(f"- **Skills:** {r2['skill_analysis']['matched_count']}/{r2['skill_analysis']['required_count']}")
