# AI-Driven Time Value of Money Solver

A chat-based actuarial engine that solves Time Value of Money problems from natural language — built for students learning the concepts and professionals who want a fast, verified answer.

**Author:** Sarah Comfort Samson
**Course:** Agentic AI Applications in Financial Mathematics — Sri Sathya Sai Institute of Actuaries
**Live app:** Live app: [n5bybqkrupkngmuz5tyq3d.streamlit.app](https://n5bybqkrupkngmuz5tyq3d.streamlit.app/)

---

## What it does

Type a Time Value of Money question in plain English — a loan, a rate conversion, a force-of-interest problem, or an irregular cash flow comparison — and the app walks through the reasoning, shows the formula, plugs in your actual numbers, and gives a verified answer, the way a tutor would on paper.

It also handles **multi-part questions** (e.g. "a. ... b. ...") by splitting them into independent sub-questions and answering each one in full.

## Concepts covered

- Present Value (PV), Future Value (FV), Payment (PMT), number of periods (n), and periodic interest rate — for loans, investments, and level annuity streams (ordinary and due)
- Rate conversions between nominal, effective, periodic, and force-of-interest bases
- Variable force of interest δ(t), including symbolic integration for time-dependent rates
- Equation of value for irregular, multi-cash-flow problems (checking whether a loan or transaction is fair/balanced)

## Architecture

```
Chat interface (app.py)
        │
        ▼
Question splitter — breaks multi-part questions into independent sub-questions
        │
        ▼
Classifier — routes each sub-question to one of four specialist subagents
        │
        ├── TVM/Annuity subagent ──────────┐
        ├── Rate Conversion subagent ──────┤
        ├── Force of Interest subagent ────┼──▶ tvm_tools.py (deterministic math)
        └── Equation of Value subagent ────┘
        │
        ▼
Formatted, tutor-style markdown answer
```

| File | Responsibility |
|---|---|
| `app.py` | Streamlit chat interface — message history, sidebar question history, example prompts, theming |
| `tvm_agent.py` | Two-stage Gemini pipeline: splits multi-part questions, classifies each one, extracts structured parameters via a type-specific Pydantic schema, and formats the final answer |
| `tvm_tools.py` | Pure, deterministic Python math — no LLM involved in any calculation. Includes a 10-check golden set validated against hand-verified FM answers |

**Design principle:** the LLM only ever *reads* the question and *extracts* parameters into a structured schema. Every actual calculation happens in plain Python (`tvm_tools.py`), so answers are reproducible and auditable rather than generated.

## Sign convention

Outflows (money paid out — deposits, loans granted, premiums paid) are **negative**. Inflows (money received — loan proceeds, withdrawals, benefit payments) are **positive**. This is enforced explicitly in every extraction prompt and explained in plain language alongside every answer.

## Tech stack

- **Frontend:** Streamlit (chat interface, session-based history)
- **LLM:** Google Gemini 2.5 Flash, via structured JSON output against Pydantic schemas
- **Math:** Custom Python (`tvm_tools.py`), SymPy for safe symbolic integration
- **Deployment:** Streamlit Community Cloud

## Running locally

```bash
git clone https://github.com/Sarah-Samson/tvm-ai-solver.git
cd tvm-ai-solver
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-api-key-here
```

Then run:

```bash
streamlit run app.py
```

## Testing

`tvm_tools.py` includes a golden set of 10 hand-verified checks covering every calculation type:

```bash
python tvm_tools.py
```

## Example questions

- *"Calculate the monthly payment for a 30-year loan of $300,000 at 6% nominal compounded monthly."*
- *"Convert a nominal interest rate of 8% compounded quarterly to an effective annual rate."*
- *"Find the accumulation factor from t=0 to t=2 if the force of interest is δ(t) = 0.02 + 0.01t."*
- *"A borrower receives $5,000 today and repays $2,000 at the end of year 1 and $3,200 at the end of year 2. Is this loan fair at a 5% effective annual rate?"*
