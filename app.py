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
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Manrope', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #101a2e 0%, #0b0f1a 45%, #05070c 100%);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .fw-hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 20px;
        padding: 34px 40px;
        margin-bottom: 28px;
        box-shadow: 0 20px 45px -20px rgba(99,102,241,0.55);
    }
    .fw-hero h1 { color: white; font-size: 2.1rem; font-weight: 800; margin: 0 0 6px 0; }
    .fw-hero p { color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0; }

    .fw-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }

    .fw-disclaimer {
        background: rgba(236,72,153,0.10);
        border: 1px solid rgba(236,72,153,0.35);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #fbcfe8;
        margin-bottom: 18px;
    }

    .fw-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .fw-priority {
        background: rgba(99,102,241,0.10);
        border-left: 3px solid #818cf8;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 14px 16px 10px 16px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.4rem;
    }

    .fw-step {
        color: rgba(255,255,255,0.55);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
