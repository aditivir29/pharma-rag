import streamlit as st

from src.retrieve import retrieve, detect_drugs, detect_intent
from src.evidence_validator import validate_evidence
from src.answer_generator import generate_answer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PharmaRAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f7f8fc;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 15px;
        background: linear-gradient(
            135deg,
            #7c3aed,
            #ec4899
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.25);
    }

    .brand-title {
        font-size: 42px;
        font-weight: 800;
        color: #172033;
        letter-spacing: -1.5px;
    }

    .tagline {
        color: #667085;
        font-size: 16px;
        margin-left: 67px;
        margin-bottom: 32px;
    }

    /* ---------- Hero ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #171c35 0%,
            #28214f 55%,
            #3d235c 100%
        );
        border-radius: 22px;
        padding: 34px 38px;
        margin-bottom: 28px;
        box-shadow: 0 15px 40px rgba(23, 32, 51, 0.15);
    }

    .hero-title {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #d8dced;
        font-size: 15px;
        line-height: 1.7;
        max-width: 780px;
    }

    .hero-pills {
        display: flex;
        gap: 10px;
        margin-top: 20px;
        flex-wrap: wrap;
    }

    .pill {
        padding: 7px 13px;
        border-radius: 20px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        color: #e8e8f2;
        font-size: 12px;
    }

    /* ---------- Search ---------- */

    .search-label {
        color: #172033;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    div[data-testid="stTextInput"] input {
        height: 52px;
        border-radius: 12px;
        border: 1px solid #d9dce5;
        background: white;
        font-size: 16px;
        padding-left: 16px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #7c3aed;
        box-shadow: 0 0 0 1px #7c3aed;
    }

    /* ---------- Button ---------- */

    div.stButton > button {
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            135deg,
            #7c3aed,
            #9f4de8
        );
        color: white;
        font-size: 15px;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.20);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 25px rgba(124, 58, 237, 0.28);
    }

    /* ---------- Section titles ---------- */

    .section-title {
        color: #172033;
        font-size: 23px;
        font-weight: 750;
        margin-top: 34px;
        margin-bottom: 15px;
    }

    /* ---------- Metric cards ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e6e8ef;
        border-radius: 14px;
        padding: 17px 20px;
        min-height: 100px;
        box-shadow: 0 4px 15px rgba(16, 24, 40, 0.04);
    }

    .metric-label {
        color: #667085;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-weight: 600;
    }

    .metric-value {
        color: #172033;
        font-size: 25px;
        font-weight: 750;
        margin-top: 7px;
    }

    /* ---------- Answer ---------- */

    .answer-card {
        background: white;
        border: 1px solid #e4e6ee;
        border-left: 5px solid #7c3aed;
        border-radius: 15px;
        padding: 24px 27px;
        box-shadow: 0 6px 22px rgba(16, 24, 40, 0.05);
        color: #252b3a;
        font-size: 17px;
        line-height: 1.75;
    }

    .answer-label {
        color: #7c3aed;
        font-size: 12px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 9px;
    }

    /* ---------- Evidence ---------- */

    .evidence-summary {
        background: white;
        border: 1px solid #e4e6ee;
        border-radius: 14px;
        padding: 17px 20px;
        margin-bottom: 10px;
    }

    .evidence-number {
        color: #7c3aed;
        font-weight: 750;
        font-size: 13px;
    }

    .evidence-title {
        color: #172033;
        font-size: 15px;
        font-weight: 650;
        margin-top: 4px;
    }

    .evidence-meta {
        color: #667085;
        font-size: 12px;
        margin-top: 7px;
    }

    .direct-badge {
        display: inline-block;
        background: #ecfdf3;
        color: #027a48;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }

    .supporting-badge {
        display: inline-block;
        background: #f2f4f7;
        color: #475467;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }

    /* ---------- Source ---------- */

    .source-link {
        display: inline-block;
        margin-top: 10px;
        padding: 8px 13px;
        border-radius: 8px;
        background: #f4f0ff;
        color: #6941c6 !important;
        text-decoration: none;
        font-size: 13px;
        font-weight: 650;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #171c35;
    }

    section[data-testid="stSidebar"] * {
        color: #e7e8f0;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        color: #aeb3c8 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .step {
        border-left: 2px solid #6d4bd8;
        padding: 5px 0 5px 13px;
        margin: 14px 0;
    }

    .step-title {
        font-size: 13px;
        font-weight: 700;
    }

    .step-text {
        color: #aeb3c8 !important;
        font-size: 11px;
        margin-top: 3px;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 12px;
        margin-top: 55px;
        padding-top: 20px;
        border-top: 1px solid #e4e6ee;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">How PharmaRAG works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'A retrieval-augmented question answering pipeline '
        'built around FDA drug-label evidence.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    steps = [
        (
            "01",
            "Drug Detection",
            "Identifies drug names in the question."
        ),
        (
            "02",
            "Intent Detection",
            "Determines interaction, warning, adverse reaction, etc."
        ),
        (
            "03",
            "Semantic Retrieval",
            "FAISS retrieves relevant FDA label passages."
        ),
        (
            "04",
            "Evidence Validation",
            "Checks whether retrieved evidence actually supports the question."
        ),
        (
            "05",
            "Answer Generation",
            "Generates a concise evidence-grounded response."
        ),
        (
            "06",
            "Citation",
            "Shows the FDA label evidence and source."
        )
    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="step">
                <div class="step-title">{number} · {title}</div>
                <div class="step-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-subtitle">'
        '<b>Knowledge source</b><br>'
        'FDA drug labeling data'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">💊</div>
        <div class="brand-title">PharmaRAG</div>
    </div>

    <div class="tagline">
        Evidence-grounded question answering over FDA drug labels
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Ask questions. Retrieve evidence. Verify the answer.
        </div>

        <div class="hero-text">
            PharmaRAG retrieves relevant FDA drug-label passages,
            validates whether the evidence supports the question,
            and generates an answer grounded in those sources.
            When sufficient evidence is unavailable, the system abstains.
        </div>

        <div class="hero-pills">
            <div class="pill">FDA Labels</div>
            <div class="pill">FAISS Retrieval</div>
            <div class="pill">Evidence Validation</div>
            <div class="pill">Local LLM</div>
            <div class="pill">Source Citations</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SEARCH
# =========================================================

st.markdown(
    '<div class="search-label">Ask PharmaRAG</div>',
    unsafe_allow_html=True
)

query = st.text_input(
    "Question",
    placeholder="e.g. Can I take ibuprofen with warfarin?",
    label_visibility="collapsed"
)

ask = st.button(
    "Ask PharmaRAG",
    use_container_width=True
)


# =========================================================
# PROCESS QUERY
# =========================================================

if ask:

    if not query.strip():

        st.warning("Please enter a question.")

        st.stop()

    with st.spinner(
        "Retrieving and validating FDA evidence..."
    ):

        try:

            drugs = detect_drugs(query)

            intent = detect_intent(query)

            results = retrieve(
                query,
                final_top_k=5
            )

            validation_result = validate_evidence(
                query=query,
                drugs=drugs,
                intent=intent,
                results=results
            )

            answer = generate_answer(
                query=query,
                validation_result=validation_result,
                drugs=drugs,
                intent=intent
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )

            st.stop()


    # =====================================================
    # QUERY ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">Query Analysis</div>',
        unsafe_allow_html=True
    )

    confidence = validation_result.get(
        "confidence",
        "none"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Drugs Detected</div>
                <div class="metric-value">{len(drugs)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Intent</div>
                <div class="metric-value">
                    {intent.replace("_", " ").title()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Evidence Confidence</div>
                <div class="metric-value">
                    {confidence.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if drugs:

        st.markdown(
            f"""
            <div style="
                color:#667085;
                font-size:13px;
                margin-top:10px;
            ">
                Detected drugs: <b>{", ".join(drugs)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # ANSWER
    # =====================================================

    st.markdown(
        '<div class="section-title">Answer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">
                Evidence-grounded response
            </div>
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    st.markdown(
        '<div class="section-title">Evidence Validation</div>',
        unsafe_allow_html=True
    )

    sufficient = validation_result.get(
        "sufficient",
        False
    )

    if sufficient:

        st.success(
            "Sufficient FDA evidence found."
        )

    else:

        st.warning(
            "Insufficient FDA evidence. "
            "The system abstained rather than generating an unsupported claim."
        )

    reason = validation_result.get(
        "reason",
        ""
    )

    if reason:

        st.caption(reason)


    # =====================================================
    # EVIDENCE
    # =====================================================

    if sufficient:

        st.markdown(
            '<div class="section-title">FDA Evidence Used</div>',
            unsafe_allow_html=True
        )

        evidence_items = validation_result.get(
            "evidence",
            []
        )

        for i, item in enumerate(
            evidence_items,
            start=1
        ):

            result = item["result"]
            chunk = result["chunk"]

            drug = chunk.get(
                "drug",
                "Unknown"
            )

            brand = chunk.get(
                "brand",
                "Unknown"
            )

            section = chunk.get(
                "section",
                "Unknown"
            )

            strength = item.get(
                "strength",
                "unknown"
            )

            text = chunk.get(
                "text",
                ""
            )

            source_url = chunk.get(
                "source_url",
                ""
            )

            if strength == "direct":

                badge = (
                    '<span class="direct-badge">'
                    'DIRECT EVIDENCE'
                    '</span>'
                )

            else:

                badge = (
                    '<span class="supporting-badge">'
                    'SUPPORTING'
                    '</span>'
                )

            with st.expander(
                f"Evidence {i}  ·  {drug}  ·  {section}"
            ):

                st.markdown(
                    f"""
                    <div class="evidence-summary">

                        <div class="evidence-number">
                            EVIDENCE {i}
                        </div>

                        <div class="evidence-title">
                            {drug} · {brand}
                        </div>

                        <div class="evidence-meta">
                            FDA Section: <b>{section}</b>
                            &nbsp;&nbsp; {badge}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "**FDA Label Evidence**"
                )

                st.write(text)

                if source_url:

                    st.markdown(
                        f"""
                        <a class="source-link"
                           href="{source_url}"
                           target="_blank">
                           ↗ Open FDA Source
                        </a>
                        """,
                        unsafe_allow_html=True
                    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        PharmaRAG · Evidence-grounded FDA drug-label QA
        <br>
        Retrieval · Validation · Local LLM · Source Attribution
    </div>
    """,
    unsafe_allow_html=True
)