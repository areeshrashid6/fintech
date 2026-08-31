"""
config.py
Central place for form options, default settings, and constants.
No AI or calculation logic lives here.
"""

APP_TITLE = "FinWise AI"
APP_TAGLINE = "Smart Budget Assistant — Educational Prototype"

DISCLAIMER = (
    "FinWise AI is an educational prototype. It does not provide guaranteed "
    "investment advice, cannot execute financial transactions, and is not "
    "connected to any real bank account. Nothing shown here is financial "
    "advice — please consult a qualified financial professional for real "
    "decisions."
)

# --- Expense categories (10, per assignment functional requirements) -------
EXPENSE_CATEGORIES = [
    ("housing", "Housing / Rent"),
    ("food", "Food & Groceries"),
    ("transportation", "Transportation"),
    ("utilities", "Utilities"),
    ("education", "Education"),
    ("healthcare", "Healthcare"),
    ("entertainment", "Entertainment"),
    ("insurance", "Insurance"),
    ("loan_debt", "Loan / Debt Payments"),
    ("other", "Other"),
]

FINANCIAL_GOALS = [
    "Save money",
    "Build an emergency fund",
    "Pay off debt",
    "Save for a vacation",
    "Start a business",
    "Improve budgeting habits",
]

CURRENCIES = ["USD", "EUR", "GBP", "PKR", "INR", "AED", "CAD", "AUD"]

# --- Model settings ----------------------------------------------------
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4

# --- Score bands (educational only) -------------------------------------
SCORE_BANDS = [
    (80, 100, "Strong", "#16a34a"),
    (60, 79, "Generally Healthy", "#2563eb"),
    (40, 59, "Needs Improvement", "#d97706"),
    (0, 39, "High Attention", "#dc2626"),
]

CACHE_OPTIONS = ["No cache", "In-memory cache", "SQLite cache (persistent)"]
