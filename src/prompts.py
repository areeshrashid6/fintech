"""
prompts.py
All prompt engineering lives here: the system role, the safety rules,
the reusable PromptTemplate (single-string), and the ChatPromptTemplate
(System + Human messages) used for the structured JSON analysis and for
the streamed narrative recommendation.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# ---------------------------------------------------------------------------
# System role + safety rules (used by both templates)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are FinWise AI, an educational personal-finance assistant.

Safety rules you must always follow:
- You are NOT a licensed financial advisor. Never claim a guaranteed outcome.
- Never suggest specific investment products, stocks, or trades.
- Never claim to execute, initiate, or connect to any real transaction or bank account.
- Frame every suggestion as educational, general guidance — not personalized financial advice.
- Always keep tone supportive, clear, and non-judgmental.
- Base your analysis ONLY on the numbers provided to you.

Your job is to analyse the user's monthly income, expenses, savings, and stated
financial goal, and produce structured, educational budgeting insight."""

JSON_SCHEMA_INSTRUCTIONS = """Return ONLY valid JSON matching exactly this schema
(no markdown fences, no commentary outside the JSON object):

{{
  "financial_summary": "",
  "financial_health_score": 0,
  "spending_analysis": [
    {{ "category": "", "observation": "", "recommendation": "" }}
  ],
  "risk_level": "",
  "top_priorities": [],
  "budget_recommendations": [],
  "savings_strategy": [],
  "next_month_action_plan": []
}}

Rules:
- "financial_health_score" must be an integer 0-100, consistent with the preliminary
  score provided (you may adjust slightly based on qualitative factors).
- "risk_level" must be one of: "LOW", "MEDIUM", "HIGH".
- "spending_analysis" should cover the 2-4 most notable expense categories.
- All arrays should contain short, concrete, plain-language bullet strings.
"""

# ---------------------------------------------------------------------------
# 1) Reusable single-string PromptTemplate carrying the financial variables
#    Required variables: monthly_income, total_expenses, remaining_income,
#    savings, savings_ratio, expense_ratio, financial_goal, expense_breakdown
# ---------------------------------------------------------------------------
FINANCIAL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "monthly_income",
        "total_expenses",
        "remaining_income",
        "savings",
        "savings_ratio",
        "expense_ratio",
        "financial_goal",
        "expense_breakdown",
        "preliminary_score",
        "currency",
    ],
    template="""User's monthly financial snapshot ({currency}):
- Monthly income: {monthly_income}
- Total expenses: {total_expenses}
- Remaining income after expenses: {remaining_income}
- Current monthly savings: {savings}
- Savings ratio: {savings_ratio}%
- Expense ratio: {expense_ratio}%
- Rule-based preliminary health score: {preliminary_score}/100
- Stated financial goal: {financial_goal}
- Expense breakdown by category: {expense_breakdown}

Analyse this snapshot and produce the structured educational insight described
in your instructions.""",
)

# ---------------------------------------------------------------------------
# 2) ChatPromptTemplate combining system instructions + safety rules +
#    dynamically inserted user data, for the structured JSON analysis chain.
# ---------------------------------------------------------------------------
JSON_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n\n" + JSON_SCHEMA_INSTRUCTIONS),
        ("human", FINANCIAL_PROMPT_TEMPLATE.template),
    ]
)

# ---------------------------------------------------------------------------
# 3) ChatPromptTemplate used for the streamed, human-readable narrative
#    recommendation shown live in the UI via .stream() / st.write_stream().
# ---------------------------------------------------------------------------
NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT
            + "\n\nWrite a warm, encouraging 120-180 word narrative recommendation "
            "in plain paragraphs (no JSON, no headers, no markdown lists) that a "
            "person could read as a short letter about their budget. End with one "
            "sentence reminding them this is educational, not professional advice.",
        ),
        ("human", FINANCIAL_PROMPT_TEMPLATE.template),
    ]
)
