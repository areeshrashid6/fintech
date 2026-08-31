"""
utils.py
Safe JSON parsing and small formatting helpers shared by app.py.
"""

import json
import re


FALLBACK_RESULT = {
    "financial_summary": "We couldn't parse a structured response this time — please try submitting again.",
    "financial_health_score": 0,
    "spending_analysis": [],
    "risk_level": "MEDIUM",
    "top_priorities": [],
    "budget_recommendations": [],
    "savings_strategy": [],
    "next_month_action_plan": [],
}


def safe_json_parse(raw_text: str) -> dict:
    """
    Safely parse the LLM's JSON output. Strips markdown fences if present,
    extracts the first {...} block as a fallback, and returns a safe
    default structure if parsing still fails — the UI must never crash
    on a malformed model response.
    """
    if not raw_text:
        return dict(FALLBACK_RESULT)

    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    result = dict(FALLBACK_RESULT)
    result["financial_summary"] = (
        "The AI response could not be parsed as JSON. Raw output has been "
        "preserved below for reference."
    )
    result["_raw"] = raw_text
    return result


def format_currency(amount: float, currency: str = "USD") -> str:
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "PKR": "Rs ", "INR": "₹", "AED": "AED ", "CAD": "C$", "AUD": "A$"}
    symbol = symbols.get(currency, currency + " ")
    try:
        return f"{symbol}{amount:,.2f}"
    except (ValueError, TypeError):
        return f"{symbol}{amount}"
