# FinWise AI — Smart Budget Assistant
Demo:
https://fintech-9h4qumpsksshphkyrljzfo.streamlit.app/

An educational LangChain + Streamlit prototype that turns a user's monthly
income, expenses, and savings into a structured, AI-generated budget dashboard.

> ⚠️ **Educational prototype only.** No real advice, no real transactions, no
> real bank connections. Every screen shows a disclaimer and directs users to
> a qualified financial professional for real decisions.

## ✨ What it does

1. **Step 1 — Connect**: the user enters their own OpenAI API key and picks a
   model/temperature. The key lives only in `st.session_state` for that
   browser session — it is never written to disk.
2. **Step 2 — Financial form**: income, 10 expense categories, current
   savings, financial goal, and currency, organized with tabs, columns, and
   an expander.
3. **Step 3 — Dashboard**: Python computes deterministic ratios and a
   preliminary score; a LangChain `ChatOpenAI` call returns structured JSON
   (summary, health score, spending analysis, risk level, priorities, budget
   tips, savings strategy, action plan); a second call **streams** a
   human-readable recommendation live into the page.

## 🗂 Project structure

```
finwise_ai/
├── app.py                     # Streamlit UI — run this
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── config.py               # form options + settings (no logic)
    ├── prompts.py               # PromptTemplate + ChatPromptTemplate + JSON schema
    ├── financial_calculator.py  # deterministic maths — no AI
    ├── chains.py                 # ChatOpenAI, chain, streaming, message demo
    ├── cache_manager.py          # in-memory + SQLite caching
    └── utils.py                  # safe JSON parsing + formatting helpers
```

## 🚀 Setup

```bash
cd finwise_ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your key into .env (optional —
                                 # you can also paste it directly in the app)
streamlit run app.py
```

You do **not** need to put your key in `.env` — the app's first screen asks
for it directly and only your key is what's used for that session's calls.
`.env` is only a convenience for local development and is git-ignored.

## 🧠 Python calculations vs. AI insight

| Layer | Does what | File |
|---|---|---|
| **Python (deterministic)** | `total_expenses`, `remaining_income`, `savings_ratio`, `expense_ratio`, and a rule-based `preliminary_score` — same inputs always give the same outputs | `financial_calculator.py` |
| **LangChain + LLM (generative)** | Turns those numbers into a written summary, a possibly-refined health score, spending observations, risk level, and actionable, plain-language suggestions | `chains.py` + `prompts.py` |

The two are kept in separate modules on purpose: the numbers you can trust
(and unit-test) never depend on the model, and the model is only ever asked
to *interpret*, never to *compute*.

## 🗄 Caching

Set from the sidebar, applied via `cache_manager.setup_cache()`:

| | In-memory | SQLite |
|---|---|---|
| Stored in | RAM | `.finwise_cache.db` file |
| Survives restart | No | Yes |
| Best for | One session | Reusing across sessions |

`set_llm_cache(...)` registers **one global cache**; LangChain checks it
before every model call, so an identical prompt returns instantly and makes
no new API request (faster + cheaper).

## 🧪 Test scenarios

| # | Input | Expect |
|---|---|---|
| 1 | Income 8000, expenses ~2000 | High score, LOW risk, growth-focused tips |
| 2 | Income 2000, expenses ~2600 | Low score, HIGH risk, urgent cost-cutting |
| 3 | Income 5000, debt 2500 | MEDIUM/HIGH risk, debt-reduction priorities |
| 4 | Income 4000, savings 1200 | High score, LOW risk, reinforce good habits |
| 5 | Income 3000, expenses 3000 | Remaining = 0, MEDIUM/HIGH risk |

## 📎 Notes

- `financial_calculator.py` guards against divide-by-zero when income is 0.
- `utils.safe_json_parse()` strips markdown fences, extracts the first
  `{...}` block, and falls back to a safe default structure if the model
  ever returns malformed JSON — the dashboard never crashes on a bad response.
- The "Developer view" expander on the dashboard shows the raw
  `SystemMessage` / `HumanMessage` / `AIMessage` objects for grading/demo
  purposes.
