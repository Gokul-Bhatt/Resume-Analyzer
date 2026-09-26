import io
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from model import calculate_ats_score
from utils import clean_text

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ResuMatch — ATS Score Predictor",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Light clean CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* White background */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
}
[data-testid="stHeader"] { background: #ffffff; }
[data-testid="stSidebar"] { background: #f8f8f8; }

/* Remove default padding */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px !important;
}

/* Buttons */
div[data-testid="stButton"] button {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="stButton"] button:hover {
    background-color: #333333 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #cccccc !important;
    border-radius: 8px !important;
    background: #fafafa !important;
    padding: 8px !important;
}

/* Textarea */
textarea {
    border: 1px solid #cccccc !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    background: #fafafa !important;
}

/* Divider */
hr { border: none; border-top: 1px solid #eeeeee; margin: 24px 0; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e8e8e8 !important;
    border-radius: 8px !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #1a1a1a !important;
}

/* Success/warning/error boxes */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Tab style */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
::-webkit-scrollbar-thumb { background: #cccccc; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── PDF Reader ─────────────────────────────────────────────────
def read_pdf(uploaded_file):
    """
    Reads a PDF file using pypdf.
    seek(0) + io.BytesIO fixes the Streamlit file buffer bug.
    """
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip(), len(reader.pages)
    except Exception as e:
        return "", 0


# ── Matplotlib Charts ──────────────────────────────────────────

def draw_score_gauge(score):
    """Draws a half-circle gauge chart for the ATS score."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Background arc
    theta = np.linspace(0, np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#eeeeee', linewidth=20, solid_capstyle='round')

    # Score arc
    score_angle = np.pi * (score / 100)
    theta_fill = np.linspace(0, score_angle, 300)
    if score >= 75:   color = '#2ecc71'
    elif score >= 55: color = '#3498db'
    elif score >= 35: color = '#f39c12'
    else:             color = '#e74c3c'
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=20, solid_capstyle='round')

    # Score text
    ax.text(0, 0.15, f"{score}%", ha='center', va='center',
            fontsize=32, fontweight='bold', color='#1a1a1a',
            fontfamily='DejaVu Sans')

    if score >= 75:   label = "Excellent"
    elif score >= 55: label = "Good"
    elif score >= 35: label = "Average"
    else:             label = "Low Match"
    ax.text(0, -0.2, label, ha='center', va='center',
            fontsize=13, color='#666666', fontfamily='DejaVu Sans')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def draw_bar_chart(breakdown):
    """Draws a horizontal bar chart for score breakdown."""
    labels = [item['label'] for item in breakdown]
    values = [item['value'] for item in breakdown]
    colors = []
    for v in values:
        if v >= 70:   colors.append('#2ecc71')
        elif v >= 45: colors.append('#3498db')
        elif v >= 25: colors.append('#f39c12')
        else:         colors.append('#e74c3c')

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor='none')

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val}%', va='center', ha='left',
                fontsize=11, color='#333333', fontfamily='DejaVu Sans')

    # Background bars
    for i, label in enumerate(labels):
        ax.barh([label], [100], color='#f0f0f0', height=0.5, edgecolor='none', zorder=0)

    ax.set_xlim(0, 115)
    ax.set_xlabel('Score (%)', fontsize=11, color='#666666')
    ax.tick_params(axis='y', labelsize=11, colors='#333333')
    ax.tick_params(axis='x', labelsize=10, colors='#888888')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#eeeeee')
    ax.xaxis.grid(True, color='#f5f5f5', linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def draw_keyword_pie(matched_count, missing_count):
    """Draws a pie chart of matched vs missing keywords."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    if matched_count == 0 and missing_count == 0:
        ax.text(0.5, 0.5, 'No keywords found', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='#888888')
        ax.axis('off')
        return fig

    sizes  = [matched_count, missing_count]
    labels = [f'Matched\n{matched_count}', f'Missing\n{missing_count}']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.04, 0.04)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 11, 'fontfamily': 'DejaVu Sans', 'color': '#333333'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color('white')
        at.set_fontweight('bold')

    ax.set_title('Keyword Coverage', fontsize=12, color='#333333',
                 fontfamily='DejaVu Sans', pad=10)
    plt.tight_layout()
    return fig


def draw_comparison_bar(score):
    """Draws a comparison bar chart showing this score vs good/average benchmarks."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    categories = ['Poor\nthreshold', 'Average\nthreshold', 'Good\nthreshold', 'Your\nresume']
    values     = [35, 55, 75, score]
    colors     = ['#e74c3c', '#f39c12', '#2ecc71',
                  '#2ecc71' if score >= 75 else '#3498db' if score >= 55 else '#f39c12' if score >= 35 else '#e74c3c']

    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333333')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=11, color='#666666')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#eeeeee')
    ax.spines['bottom'].set_color('#eeeeee')
    ax.tick_params(axis='both', colors='#888888', labelsize=10)
    ax.yaxis.grid(True, color='#f5f5f5', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('Your Score vs Benchmarks', fontsize=12,
                 color='#333333', pad=10)
    plt.tight_layout()
    return fig


# ── Score Breakdown Calculator ─────────────────────────────────
def get_breakdown(resume_text, jd_text, matched, missing):
    r_words = set(clean_text(resume_text).split())
    j_words = set(clean_text(jd_text).split())
    total   = len(j_words) if j_words else 1

    kw_score  = round((len(matched) / (len(matched) + len(missing))) * 100) if (matched or missing) else 0
    cov_score = round((len(r_words & j_words) / total) * 100)
    len_score = min(round((len(r_words) / 300) * 100), 100)
    dens_score= min(round((len(matched) / max(len(r_words), 1)) * 300), 100)

    return [
        {"label": "Keyword match",    "value": kw_score,   "hint": "% of JD keywords in resume"},
        {"label": "Word coverage",    "value": cov_score,  "hint": "Unique JD words covered"},
        {"label": "Resume detail",    "value": len_score,  "hint": "Resume length & richness"},
        {"label": "Skill density",    "value": dens_score, "hint": "Matched skills per word"},
    ]


#  PAGE LAYOUT************************************************

# ── Sidebar — About ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About this tool")
    st.markdown("""
This tool checks how well your resume matches a job description using:

**Step 1** — PDF text extraction using pypdf

**Step 2** — Text cleaning using NLTK (remove stopwords)

**Step 3** — Word vectorisation using Scikit-learn CountVectorizer

**Step 4** — Cosine similarity scoring

---
**Tech Stack**
- Python 3.11
- Streamlit
- pypdf
- NLTK
- Scikit-learn
- Matplotlib

---
*School of Computer Applications*
*Graphic Era Hill University*
*Dehradun*
    """)

# ── Header ─────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("## ResuMatch — ATS Score Predictor")
    st.markdown(
        '<p style="color:#888888; font-size:14px; margin-top:-12px;">'
        'Upload your resume and paste a job description to get your ATS compatibility score</p>',
        unsafe_allow_html=True
    )
st.divider()

# ── Input Section ──────────────────────────────────────────────
col_in1, col_in2 = st.columns(2, gap="large")

with col_in1:
    st.markdown("**Step 1 — Upload Resume PDF**")
    st.caption("Use a text-based PDF (made in Word or Google Docs)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], label_visibility="collapsed")
    if resume_file:
        st.success(f"✓  {resume_file.name} loaded")

with col_in2:
    st.markdown("**Step 2 — Paste Job Description**")
    st.caption("Copy the full job description from the job portal")
    job_desc = st.text_area(
        "Job Description", height=165, label_visibility="collapsed",
        placeholder="Paste job description here...\n\nExample:\nWe are looking for a Python developer with experience in Machine Learning, Scikit-learn, NLP, TF-IDF, Pandas, SQL..."
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
run = st.button("Analyse Resume →", use_container_width=True)




if run:
    # ── Validate ───────────────────────────────────────────────
    if not resume_file:
        st.warning("Please upload your resume PDF first.")
        st.stop()
    if not job_desc.strip():
        st.warning("Please paste the job description.")
        st.stop()
    if len(job_desc.strip().split()) < 15:
        st.warning("Job description seems too short. Paste the full JD.")
        st.stop()

    # ── Extract PDF ────────────────────────────────────────────
    with st.spinner("Reading PDF..."):
        resume_text, num_pages = read_pdf(resume_file)

    if not resume_text:
        st.error("Could not read text from this PDF. Please use a text-based PDF (not a scanned image).")
        st.stop()

    # ── Calculate Score ────────────────────────────────────────
    with st.spinner("Calculating ATS score..."):
        score, matched, missing = calculate_ats_score(resume_text, job_desc)
        breakdown = get_breakdown(resume_text, job_desc, matched, missing)

    st.divider()
    st.markdown("### Results")

    # ── Tab Layout ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "  Overview  ", "  Keywords  ", "  Charts  ", "  Recommendations  "
    ])

    # ══════════════════════════
    #  TAB 1 — OVERVIEW
    # ══════════════════════════
    with tab1:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        ov1, ov2, ov3 = st.columns([1.2, 1, 1], gap="large")

        with ov1:
            # Gauge chart
            fig_gauge = draw_score_gauge(score)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

        with ov2:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            if score >= 75:
                st.success("**Excellent Match**")
                verdict = f"Your resume strongly matches this job description. {len(matched)} out of {len(matched)+len(missing)} JD keywords were found."
            elif score >= 55:
                st.info("**Good Match**")
                verdict = f"Your resume is a reasonable match. {len(matched)} keywords found, {len(missing)} missing. A few improvements will help."
            elif score >= 35:
                st.warning("**Average Match**")
                verdict = f"Partial match. Only {len(matched)} of {len(matched)+len(missing)} keywords found. Resume needs tailoring for this role."
            else:
                st.error("**Low Match**")
                verdict = f"Your resume matches very few JD keywords ({len(matched)} found). Significant improvements needed."

            st.markdown(f'<p style="font-size:13px; color:#555555; margin-top:8px;">{verdict}</p>',
                        unsafe_allow_html=True)

        with ov3:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.metric("ATS Score",       f"{score}%")
            st.metric("Matched Keywords", len(matched))
            st.metric("Missing Keywords", len(missing))
            st.metric("Resume Pages",     num_pages)

        st.divider()

        # Breakdown metrics row
        st.markdown("**Score Breakdown**")
        bcols = st.columns(4)
        for i, item in enumerate(breakdown):
            with bcols[i]:
                st.metric(item['label'], f"{item['value']}%", help=item['hint'])
        st.progress(int(score))

    # ══════════════════════════
    #  TAB 2 — KEYWORDS
    # ══════════════════════════
    with tab2:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        kw1, kw2 = st.columns(2, gap="large")

        with kw1:
            st.markdown(f"**✅ Matched Keywords — {len(matched)}**")
            st.caption("These JD keywords were found in your resume")
            if matched:
                # Show as a styled table
                for i, kw in enumerate(matched[:30]):
                    st.markdown(
                        f'<span style="display:inline-block; background:#e8f5e9; color:#1b5e20; '
                        f'padding:3px 10px; border-radius:4px; font-size:12px; '
                        f'font-family:\'IBM Plex Mono\',monospace; margin:3px;">{kw}</span>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No keyword matches found.")

        with kw2:
            st.markdown(f"**❌ Missing Keywords — {len(missing)}**")
            st.caption("Add these to your resume to improve your score")
            if missing:
                for kw in missing[:30]:
                    st.markdown(
                        f'<span style="display:inline-block; background:#ffebee; color:#b71c1c; '
                        f'padding:3px 10px; border-radius:4px; font-size:12px; '
                        f'font-family:\'IBM Plex Mono\',monospace; margin:3px;">{kw}</span>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("All keywords matched — excellent!")

        st.divider()

        # Keyword frequency table
        st.markdown("**Top JD Keywords and their frequency in your resume**")
        r_freq = {}
        for word in clean_text(resume_text).split():
            r_freq[word] = r_freq.get(word, 0) + 1

        jd_top = sorted(
            set(clean_text(job_desc).split()),
            key=lambda w: r_freq.get(w, 0), reverse=True
        )[:20]

        freq_data = []
        for word in jd_top:
            freq = r_freq.get(word, 0)
            status = "✅ Found" if freq > 0 else "❌ Missing"
            freq_data.append({"Keyword": word, "In Resume": status, "Count in Resume": freq})

        if freq_data:
            import pandas as pd
            df = pd.DataFrame(freq_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ══════════════════════════
    #  TAB 3 — CHARTS
    # ══════════════════════════
    with tab3:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown("**Score Breakdown (Bar Chart)**")
            fig_bar = draw_bar_chart(breakdown)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

        with ch2:
            st.markdown("**Keyword Coverage (Pie Chart)**")
            fig_pie = draw_keyword_pie(len(matched), len(missing))
            st.pyplot(fig_pie, use_container_width=True)
            plt.close()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        ch3_col, ch4_col = st.columns(2, gap="large")

        with ch3_col:
            st.markdown("**Your Score vs Benchmarks**")
            fig_comp = draw_comparison_bar(score)
            st.pyplot(fig_comp, use_container_width=True)
            plt.close()

        with ch4_col:
            st.markdown("**Word Frequency — Top Resume Keywords**")
            # Top 10 resume words by frequency
            all_words = clean_text(resume_text).split()
            freq_counter = {}
            for w in all_words:
                freq_counter[w] = freq_counter.get(w, 0) + 1
            top_words = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)[:10]

            if top_words:
                words_list  = [w for w, _ in top_words]
                counts_list = [c for _, c in top_words]

                fig_freq, ax = plt.subplots(figsize=(6, 3.5))
                fig_freq.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                ax.bar(words_list, counts_list, color='#3498db', edgecolor='none', width=0.6)
                ax.set_xlabel('Keywords', fontsize=10, color='#666666')
                ax.set_ylabel('Frequency', fontsize=10, color='#666666')
                ax.tick_params(axis='x', labelsize=9, colors='#444444', rotation=30)
                ax.tick_params(axis='y', labelsize=9, colors='#888888')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#eeeeee')
                ax.spines['bottom'].set_color('#eeeeee')
                ax.yaxis.grid(True, color='#f5f5f5', linewidth=0.8)
                ax.set_axisbelow(True)
                ax.set_title('Top 10 Resume Keywords', fontsize=12, color='#333333', pad=8)
                plt.tight_layout()
                st.pyplot(fig_freq, use_container_width=True)
                plt.close()

    # ══════════════════════════
    #  TAB 4 — RECOMMENDATIONS
    # ══════════════════════════
    with tab4:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        tips = []

        # Smart tips based on analysis
        if missing:
            top5 = ", ".join(f'**{w}**' for w in missing[:5])
            tips.append({
                "icon": "🔑",
                "title": "Add missing keywords",
                "detail": f"These JD keywords are absent from your resume: {top5}{'...' if len(missing) > 5 else ''}. Add them naturally in your skills or project sections."
            })

        if breakdown[2]['value'] < 60:
            tips.append({
                "icon": "📝",
                "title": "Expand your resume",
                "detail": "Your resume seems short. Add more detail to your project descriptions — mention the tools used, what you built, and what the outcome was."
            })

        if breakdown[0]['value'] < 50:
            tips.append({
                "icon": "🎯",
                "title": "Tailor this resume to the job",
                "detail": "Less than half the JD keywords appear in your resume. Consider rewriting your skills section and project descriptions to use the same terminology as the job description."
            })

        tips.append({
            "icon": "✏️",
            "title": "Use exact words from the JD",
            "detail": "ATS systems match exact keywords, not synonyms. If the JD says 'Scikit-learn', write 'Scikit-learn' in your resume — not 'sklearn' or 'machine learning library'."
        })

        tips.append({
            "icon": "📋",
            "title": "Add a dedicated Skills section",
            "detail": "Put all your technical tools and languages in one clearly labelled section. This makes it easy for ATS to find and match them."
        })

        tips.append({
            "icon": "📂",
            "title": "Describe each project with tool names",
            "detail": "For each project, mention the exact tools you used. Example: 'Built a classifier using Python, Scikit-learn, and Pandas.'"
        })

        if score >= 75:
            tips.append({
                "icon": "✅",
                "title": "You are in good shape!",
                "detail": "Your resume already matches this JD well. Focus on fine-tuning rather than major rewrites. Make sure your resume formatting is clean and ATS-friendly."
            })

        for tip in tips:
            st.markdown(
                f"""<div style="background:#fafafa; border:1px solid #e8e8e8; border-left:4px solid #1a1a1a;
                border-radius:6px; padding:14px 18px; margin-bottom:12px;">
                <div style="font-size:15px; font-weight:600; color:#1a1a1a; margin-bottom:4px;">
                {tip['icon']}  {tip['title']}</div>
                <div style="font-size:13px; color:#555555; line-height:1.6;">{tip['detail']}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.divider()

    # ── Extracted Text Preview ─────────────────────────────────
    with st.expander(f"📄 View extracted resume text ({len(resume_text.split())} words, {num_pages} page(s))"):
        st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

