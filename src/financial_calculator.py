"""
financial_calculator.py
Pure, deterministic Python math. No AI here — the same inputs will always
produce the same outputs. This is intentionally kept separate from the
LLM layer so calculations can be unit-tested and trusted independently
of any model output.
"""

from src.config import SCORE_BANDS


def compute_financials(monthly_income: float, expenses: dict, savings: float) -> dict:
    """
    Compute total expenses, remaining income, savings ratio, expense ratio,
    and a rule-based preliminary financial health score (0-100).

    expenses: dict of {category_key: amount}
    """
    monthly_income = max(float(monthly_income or 0), 0)
    savings = max(float(savings or 0), 0)
    expenses = {k: max(float(v or 0), 0) for k, v in (expenses or {}).items()}

    total_expenses = sum(expenses.values())
    remaining_income = monthly_income - total_expenses

    # Guard against divide-by-zero when income is 0
    if monthly_income > 0:
        savings_ratio = (savings / monthly_income) * 100
        expense_ratio = (total_expenses / monthly_income) * 100
        debt_ratio = (expenses.get("loan_debt", 0) / monthly_income) * 100
    else:
        savings_ratio = 0.0
        expense_ratio = 0.0
        debt_ratio = 0.0

    preliminary_score = _compute_preliminary_score(
        savings_ratio=savings_ratio,
        expense_ratio=expense_ratio,
        debt_ratio=debt_ratio,
        remaining_income=remaining_income,
        monthly_income=monthly_income,
    )

    return {
        "monthly_income": round(monthly_income, 2),
        "total_expenses": round(total_expenses, 2),
        "remaining_income": round(remaining_income, 2),
        "savings": round(savings, 2),
        "savings_ratio": round(savings_ratio, 2),
        "expense_ratio": round(expense_ratio, 2),
        "debt_ratio": round(debt_ratio, 2),
        "preliminary_score": round(preliminary_score, 1),
        "expense_breakdown": expenses,
    }


def _compute_preliminary_score(
    savings_ratio: float,
    expense_ratio: float,
    debt_ratio: float,
    remaining_income: float,
    monthly_income: float,
) -> float:
    """
    Weighted 0-100 heuristic based on:
      - savings ratio (higher is better)      -> 35%
      - leftover / remaining income sign+size  -> 25%
      - expense ratio (lower is better)        -> 25%
      - debt burden (lower is better)          -> 15%
    """
    if monthly_income <= 0:
        return 0.0

    savings_score = min(savings_ratio / 30 * 100, 100)  # 30%+ savings = full marks
    expense_score = max(0.0, 100 - expense_ratio)        # 0% expenses = full marks
    debt_score = max(0.0, 100 - debt_ratio * 2.5)         # debt burden penalized harder

    leftover_pct = (remaining_income / monthly_income) * 100
    leftover_score = max(0.0, min(leftover_pct * 2, 100))

    score = (
        savings_score * 0.35
        + leftover_score * 0.25
        + expense_score * 0.25
        + debt_score * 0.15
    )
    return max(0.0, min(score, 100.0))


def score_band(score: float) -> tuple:
    """Return (label, color) for a given score using SCORE_BANDS."""
    for low, high, label, color in SCORE_BANDS:
        if low <= score <= high:
            return label, color
    return "Unknown", "#6b7280"
