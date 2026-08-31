"""
app.py — FinWise AI
Run with: streamlit run app.py

Flow:
  Step 1  "connect"   -> user enters their OpenAI API key + model settings
  Step 2  "form"       -> user enters income, expenses, savings, goal
  Step 3  "dashboard"  -> deterministic calculations + AI-generated dashboard
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src import config
from src.financial_calculator import compute_financials, score_band
from src.prompts import JSON_CHAT_TEMPLATE
from src.chains import get_llm, run_analysis, stream_recommendations, message_demo
from src.cache_manager import setup_cache
from src.utils import safe_json_parse, format_currency

load_dotenv()

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* ==============================
   GLOBAL
   ============================== */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1220 !important;
    color: #f5f7f6 !important;
}

.main {
    background: #0b1220 !important;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ==============================
   ALL TEXT
   ============================== */

.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp li {
    color: #e7ece9;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #ffffff !important;
}

.stMarkdown {
    color: #e7ece9 !important;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #9da9a4 !important;
}

/* ==============================
   HERO
   ============================== */

.hero {
    padding: 3.2rem 3.2rem 2.8rem;
    border: 1px solid #263449;
    border-radius: 28px;
    background: linear-gradient(
        135deg,
        #111b2d 0%,
        #12251f 100%
    );
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,.25);
}

.eyebrow {
    color: #69d3ad !important;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    font-size: .78rem;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    line-height: 1;
    margin: .5rem 0 1rem;
    color: #ffffff !important;
}

.hero p {
    color: #aebbb5 !important;
    max-width: 720px;
    font-size: 1.08rem;
    line-height: 1.7;
}

/* ==============================
   CARDS
   ============================== */

.card {
    background: #121c2c !important;
    border: 1px solid #29374b;
    border-radius: 20px;
    padding: 1.35rem 1.45rem;
    height: 100%;
    box-shadow: 0 10px 30px rgba(0,0,0,.18);
}

.card p {
    color: #9facaa !important;
}

.small-label {
    color: #8f9d97 !important;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700;
}

.big-number {
    color: #ffffff !important;
    font-size: 1.85rem;
    font-weight: 700;
    margin-top: .35rem;
}

/* ==============================
   SECTION TITLES
   ============================== */

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    margin: 1.7rem 0 .8rem;
    color: #ffffff !important;
}

/* ==============================
   SIDEBAR
   ============================== */

section[data-testid="stSidebar"] {
    background: #0d1726 !important;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] * {
    color: #e8eeeb !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #29374b !important;
}

/* ==============================
   STEPS
   ============================== */

.step {
    display: flex;
    gap: 12px;
    align-items: center;
    margin: 7px 0;
    color: #899791 !important;
    font-size: .9rem;
}

.step span {
    color: inherit !important;
}

.step.active {
    color: #ffffff !important;
    font-weight: 700;
}

.step-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #202b3d;
    color: #a4b0aa !important;
    font-weight: 700;
}

.step.active .step-dot {
    background: #1f8b68;
    color: #ffffff !important;
}

/* ==============================
   FORM
   ============================== */

div[data-testid="stForm"] {
    border: 1px solid #29374b;
    border-radius: 22px;
    padding: 1.5rem;
    background: #111a29 !important;
}

/* Input labels */

[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
    color: #e8eeeb !important;
    font-weight: 600;
}

/* Input boxes */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #182335 !important;
    color: #ffffff !important;
    border-color: #34445a !important;
}

/* Placeholder */

input::placeholder {
    color: #7e8b87 !important;
}

/* Selectbox text */

[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* ==============================
   BUTTONS
   ============================== */

div.stButton > button,
button[kind="primary"],
.stFormSubmitButton button {
    background: #1f8b68 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px;
    min-height: 44px;
    font-weight: 700;
    transition: .2s ease;
}

div.stButton > button:hover,
.stFormSubmitButton button:hover {
    background: #2ca77f !important;
    color: #ffffff !important;
    border: none !important;
}

/* ==============================
   METRICS
   ============================== */

[data-testid="stMetric"] {
    background: #121c2c !important;
    border: 1px solid #29374b;
    border-radius: 18px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    color: #9da9a4 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* ==============================
   TABS
   ============================== */

button[data-baseweb="tab"] {
    color: #9da9a4 !important;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #69d3ad !important;
}

[data-baseweb="tab-highlight"] {
    background-color: #69d3ad !important;
}

/* ==============================
   EXPANDERS
   ============================== */

[data-testid="stExpander"] {
    background: #121c2c !important;
    border: 1px solid #29374b !important;
    border-radius: 16px !important;
}

[data-testid="stExpander"] summary {
    color: #ffffff !important;
}

[data-testid="stExpander"] summary p {
    color: #ffffff !important;
}

/* ==============================
   RESULT BOX
   ============================== */

.result-box {
    border: 1px solid #29374b;
    border-radius: 18px;
    padding: 1.2rem;
    background: #121c2c !important;
}

.result-box p {
    color: #c1cbc6 !important;
}

/* ==============================
   DISCLAIMER
   ============================== */

.disclaimer {
    padding: 12px 15px;
    border-radius: 12px;
    background: #292313 !important;
    border: 1px solid #66582e;
    color: #e8d99a !important;
    font-size: .82rem;
    line-height: 1.5;
}

.disclaimer * {
    color: #e8d99a !important;
}

/* ==============================
   ALERTS
   ============================== */

[data-testid="stAlert"] {
    color: #ffffff !important;
}

[data-testid="stAlert"] p {
    color: inherit !important;
}

/* ==============================
   PROGRESS BAR
   ============================== */

[data-testid="stProgress"] {
    margin-top: 10px;
}

[data-testid="stProgress"] > div {
    background: #263447 !important;
}

/* ==============================
   GENERAL LINKS
   ============================== */

a {
    color: #69d3ad !important;
}

/* ==============================
   SCROLLBAR
   ============================== */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0b1220;
}

::-webkit-scrollbar-thumb {
    background: #34445a;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "step": "connect",
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("FINWISE_DEFAULT_MODEL", config.DEFAULT_MODEL),
    "temperature": float(os.getenv("FINWISE_DEFAULT_TEMPERATURE", config.DEFAULT_TEMPERATURE)),
    "cache_type": "In-memory cache",
    "cache_status": "",
    "financials": None,
    "analysis": None,
    "currency": "USD",
    "financial_goal": config.FINANCIAL_GOALS[0],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_session():
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.step = "connect"


# ---------------------------------------------------------------------------
# Sidebar (shown once the user is past the API key step)
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 💠 FinWise AI")
        st.caption(config.APP_TAGLINE)
        st.markdown(
            f'<div class="fw-disclaimer">{config.DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Model settings**")
        st.session_state.model = st.selectbox(
            "Model", config.AVAILABLE_MODELS,
            index=config.AVAILABLE_MODELS.index(st.session_state.model)
            if st.session_state.model in config.AVAILABLE_MODELS else 0,
        )
        st.session_state.temperature = st.slider("Creativity (temperature)", 0.0, 1.0, st.session_state.temperature, 0.1)

        st.markdown("**Caching**")
        st.session_state.cache_type = st.selectbox("Cache type", config.CACHE_OPTIONS,
                                                     index=config.CACHE_OPTIONS.index(st.session_state.cache_type))
        st.session_state.cache_status = setup_cache(st.session_state.cache_type)
        st.caption(st.session_state.cache_status)

        st.divider()
        if st.button("🔄 Reset session", use_container_width=True):
            reset_session()
            st.rerun()

        st.caption("Educational prototype · not financial advice")


# ---------------------------------------------------------------------------
# Step 1 — Connect (API key)
# ---------------------------------------------------------------------------
def render_connect_step():
    st.markdown(
        f"""
        <div class="fw-hero">
            <h1>💠 {config.APP_TITLE}</h1>
            <p>{config.APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown('<div class="fw-step">Step 1 of 2</div>', unsafe_allow_html=True)
        st.markdown("### Connect your AI model")
        st.write(
            "FinWise AI uses your own OpenAI API key to generate insights — "
            "your key is kept only in this session and is never stored on disk."
        )

        with st.form("connect_form"):
            api_key_input = st.text_input(
                "OpenAI API key", value=st.session_state.api_key, type="password",
                placeholder="sk-...",
            )
            model_choice = st.selectbox("Model", config.AVAILABLE_MODELS,
                                         index=config.AVAILABLE_MODELS.index(st.session_state.model)
                                         if st.session_state.model in config.AVAILABLE_MODELS else 0)
            temp_choice = st.slider("Creativity (temperature)", 0.0, 1.0, st.session_state.temperature, 0.1)
            submitted = st.form_submit_button("Continue →", use_container_width=True)

        if submitted:
            if not api_key_input or not api_key_input.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key (starts with 'sk-').")
            else:
                st.session_state.api_key = api_key_input
                st.session_state.model = model_choice
                st.session_state.temperature = temp_choice
                st.session_state.step = "form"
                st.rerun()

        st.markdown(
            f'<div class="fw-disclaimer">{config.DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="fw-card">', unsafe_allow_html=True)
        st.markdown("#### What happens next")
        st.markdown(
            "1. Enter income, expenses, savings, and your goal\n"
            "2. Python computes your ratios and a preliminary score\n"
            "3. The AI turns those numbers into a structured, educational dashboard\n"
            "4. A written recommendation streams in live"
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Step 2 — Financial form
# ---------------------------------------------------------------------------
def render_form_step():
    st.markdown(
        f"""
        <div class="fw-hero">
            <h1>💠 Tell us about your month</h1>
            <p>Every number stays local to this session and is only used to generate your dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fw-step">Step 2 of 2</div>', unsafe_allow_html=True)

    with st.form("financial_form"):
        tab_income, tab_expenses, tab_goal = st.tabs(["💰 Income & Savings", "🧾 Expenses", "🎯 Goal"])

        with tab_income:
            col1, col2 = st.columns(2)
            with col1:
                monthly_income = st.number_input("Monthly income", min_value=0.0, step=100.0, value=5000.0)
            with col2:
                savings = st.number_input("Current monthly savings", min_value=0.0, step=50.0, value=500.0)
            currency = st.selectbox("Currency", config.CURRENCIES,
                                     index=config.CURRENCIES.index(st.session_state.currency))

        with tab_expenses:
            st.caption("Enter your monthly spend per category.")
            expenses = {}
            cols = st.columns(2)
            for i, (key, label) in enumerate(config.EXPENSE_CATEGORIES):
                with cols[i % 2]:
                    expenses[key] = st.number_input(label, min_value=0.0, step=25.0, value=0.0, key=f"exp_{key}")

        with tab_goal:
            financial_goal = st.selectbox("Financial goal", config.FINANCIAL_GOALS,
                                           index=config.FINANCIAL_GOALS.index(st.session_state.financial_goal))
            with st.expander("ℹ️ How your health score is calculated"):
                st.write(
                    "Python computes a rule-based preliminary score from your savings ratio, "
                    "leftover income, expense ratio, and debt burden — before the AI ever sees "
                    "your numbers. The AI may refine this slightly using qualitative context."
                )

        col_back, col_submit = st.columns([1, 2])
        with col_back:
            back = st.form_submit_button("← Back", use_container_width=True)
        with col_submit:
            submitted = st.form_submit_button("Generate my dashboard →", use_container_width=True, type="primary")

    if back:
        st.session_state.step = "connect"
        st.rerun()

    if submitted:
        st.session_state.currency = currency
        st.session_state.financial_goal = financial_goal
        st.session_state.financials = compute_financials(monthly_income, expenses, savings)
        st.session_state.step = "dashboard"
        st.rerun()


# ---------------------------------------------------------------------------
# Step 3 — AI dashboard
# ---------------------------------------------------------------------------
def render_dashboard_step():
    fin = st.session_state.financials
    currency = st.session_state.currency

    st.markdown(
        f"""
        <div class="fw-hero">
            <h1>📊 Your financial dashboard</h1>
            <p>Goal: {st.session_state.financial_goal} · Currency: {currency}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Financial overview (Python-calculated, deterministic) -----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly income", format_currency(fin["monthly_income"], currency))
    c2.metric("Total expenses", format_currency(fin["total_expenses"], currency))
    c3.metric("Remaining balance", format_currency(fin["remaining_income"], currency))
    c4.metric("Current savings", format_currency(fin["savings"], currency))

    # Expense breakdown as a chart, not a raw table
    st.markdown("#### Expense breakdown")
    chart_data = {label: fin["expense_breakdown"].get(key, 0)
                  for key, label in config.EXPENSE_CATEGORIES if fin["expense_breakdown"].get(key, 0) > 0}
    if chart_data:
        st.bar_chart(chart_data, horizontal=True)
    else:
        st.caption("No expenses entered yet.")

    st.divider()

    # --- Run the AI analysis once, cache result in session state ---------
    if st.session_state.analysis is None:
        with st.spinner("Analyzing your finances with FinWise AI..."):
            try:
                llm = get_llm(st.session_state.api_key, st.session_state.model, st.session_state.temperature)
                inputs = {
                    "monthly_income": fin["monthly_income"],
                    "total_expenses": fin["total_expenses"],
                    "remaining_income": fin["remaining_income"],
                    "savings": fin["savings"],
                    "savings_ratio": fin["savings_ratio"],
                    "expense_ratio": fin["expense_ratio"],
                    "financial_goal": st.session_state.financial_goal,
                    "expense_breakdown": fin["expense_breakdown"],
                    "preliminary_score": fin["preliminary_score"],
                    "currency": currency,
                }
                raw_json = run_analysis(llm, inputs)
                st.session_state.analysis = safe_json_parse(raw_json)
                st.session_state._chain_inputs = inputs
            except Exception as e:
                st.error(f"Couldn't reach the AI model: {e}")
                st.stop()

    analysis = st.session_state.analysis

    # --- AI analysis section ----------------------------------------------
    score = analysis.get("financial_health_score", fin["preliminary_score"])
    label, color = score_band(score)
    risk = analysis.get("risk_level", "MEDIUM")
    risk_colors = {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626"}

    col_score, col_risk = st.columns([2, 1])
    with col_score:
        st.markdown("#### Financial health score")
        st.progress(min(max(int(score), 0), 100) / 100)
        st.markdown(
            f'<span class="fw-badge" style="background:{color}22;color:{color};">{int(score)}/100 · {label}</span>',
            unsafe_allow_html=True,
        )
    with col_risk:
        st.markdown("#### Risk level")
        st.markdown(
            f'<span class="fw-badge" style="background:{risk_colors.get(risk, "#6b7280")}22;'
            f'color:{risk_colors.get(risk, "#6b7280")};">{risk}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="fw-card">{analysis.get("financial_summary", "")}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Priorities", "🧾 Spending analysis", "📋 Budget tips", "💵 Savings strategy", "✅ Action plan"])

    with tabs[0]:
        for p in analysis.get("top_priorities", []):
            st.markdown(f'<div class="fw-priority">{p}</div>', unsafe_allow_html=True)
        if not analysis.get("top_priorities"):
            st.caption("No priorities returned.")

    with tabs[1]:
        for item in analysis.get("spending_analysis", []):
            with st.expander(f"📌 {item.get('category', 'Category')}"):
                st.write(f"**Observation:** {item.get('observation', '')}")
                st.write(f"**Recommendation:** {item.get('recommendation', '')}")
        if not analysis.get("spending_analysis"):
            st.caption("No spending analysis returned.")

    with tabs[2]:
        for b in analysis.get("budget_recommendations", []):
            st.markdown(f"- {b}")

    with tabs[3]:
        for s in analysis.get("savings_strategy", []):
            st.markdown(f"- {s}")

    with tabs[4]:
        for a in analysis.get("next_month_action_plan", []):
            st.markdown(f"- {a}")

    st.divider()

    # --- Streamed narrative recommendation ---------------------------------
    st.markdown("#### A note from FinWise AI")
    if "narrative_done" not in st.session_state:
        st.session_state.narrative_done = False

    if not st.session_state.narrative_done:
        streaming_llm = get_llm(st.session_state.api_key, st.session_state.model,
                                 st.session_state.temperature, streaming=True)
        st.write_stream(stream_recommendations(streaming_llm, st.session_state._chain_inputs))
        st.session_state.narrative_done = True
    else:
        st.info("Streamed recommendation already generated for this session. Reset to regenerate.")

    with st.expander("🔎 Developer view — raw message demo (System / Human / AI)"):
        for m in message_demo(st.session_state._chain_inputs):
            st.code(f"{type(m).__name__}: {m.content}", language="text")

    st.markdown(
        f'<div class="fw-disclaimer">{config.DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    if st.button("↺ Start a new analysis"):
        st.session_state.analysis = None
        st.session_state.narrative_done = False
        st.session_state.step = "form"
        st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if st.session_state.step == "connect":
    render_connect_step()
else:
    render_sidebar()
    if st.session_state.step == "form":
        render_form_step()
    elif st.session_state.step == "dashboard":
        render_dashboard_step()
