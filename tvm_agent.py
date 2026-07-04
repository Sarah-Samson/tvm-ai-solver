import json
import math
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import tvm_tools

MODEL_NAME = "gemini-2.5-flash"

# ==============================================================================
# STAGE 0: QUESTION SPLITTING
# Many real assignment questions bundle several distinct asks into one
# message (e.g. "a. ... b. ... c. ..."). Everything downstream (classifier,
# extractors) assumes a single question, so this stage runs first and breaks
# a multi-part message into separate, fully self-contained questions before
# any of that happens. Each part is solved independently and the answers are
# combined at the end.
# ==============================================================================
class QuestionSplit(BaseModel):
    sub_questions: List[str] = Field(
        description=(
            "One entry per distinct question being asked. If the message contains "
            "multiple labeled parts (e.g. 'a.', 'b.', 'c.') or otherwise asks more than "
            "one distinct thing, split each into its own fully self-contained question - "
            "repeating any shared context (rates, dates, dollar amounts, time horizons) "
            "so each one can be solved entirely on its own, without seeing the others. "
            "If the message only contains one question, return a list with just that "
            "single question in it (lightly cleaned up if helpful, otherwise unchanged)."
        )
    )


SPLIT_SYSTEM_PROMPT = """
You are a question-splitting assistant for an actuarial math engine. Some messages bundle
more than one distinct question together (commonly labeled 'a.', 'b.', 'c.', or joined with
'and'/'also'). Your only job is to split these into separate, fully self-contained questions -
do not solve anything yourself, and do not extract any numbers for calculation.

RULES:
1. Each sub-question must restate all the shared context it depends on (rates, dates, dollar
   amounts, time horizons) in full, since it will be solved completely independently of the
   original message and the other parts.
2. Preserve the original order of the parts.
3. If the message only contains one question, return exactly one item.
4. Never merge two distinct asks into a single sub-question, even if they share the same
   underlying numbers.
"""


# ==============================================================================
# STAGE 1: PROBLEM CLASSIFICATION
# Decides which specialist subagent should handle the query, before any
# numeric extraction happens. This is what lets rate-conversion and
# force-of-interest questions escape the loan/annuity schema they used to
# get force-fit into.
# ==============================================================================
class ProblemClassification(BaseModel):
    problem_type: Literal[
        "tvm_annuity", "rate_conversion", "force_of_interest", "equation_of_value"
    ] = Field(
        description=(
            "tvm_annuity: solving for pv, fv, pmt, n, or rate of a loan, investment, "
            "or single LEVEL annuity/payment stream (one constant payment amount). "
            "rate_conversion: converting a rate between nominal, effective, periodic, "
            "or force-of-interest bases, with no payment stream to solve for. "
            "force_of_interest: problems that give a variable/time-dependent force of "
            "interest delta(t) and require integrating it over a time interval, with "
            "no explicit list of cash flows to value. "
            "equation_of_value: problems with two or more distinct cash flows occurring "
            "at different, explicitly stated times that must be valued/compared at a "
            "common point in time - e.g. checking whether a loan and its repayments "
            "balance, comparing two irregular payment streams, or replacing several "
            "cash flows with an equivalent single payment. Use this instead of "
            "tvm_annuity whenever the cash flows are irregular or individually listed "
            "rather than one repeating level payment."
        )
    )


CLASSIFY_SYSTEM_PROMPT = """
You are a routing classifier for an actuarial math engine with four specialist solvers.
Read the user's question and decide which ONE solver it needs. Do not attempt to solve
anything yourself, and do not extract any numbers. Only classify.
"""


# ==============================================================================
# STAGE 2A: TVM ANNUITY EXTRACTION (loans, investments, level payment streams)
# ==============================================================================
class TVMParameters(BaseModel):
    calc_type: Literal["pv", "fv", "pmt", "nper", "rate"] = Field(
        description="The unknown target variable to solve for."
    )
    pv: float = Field(
        default=0.0,
        description="Present Value. Cash outflows (investments/loans granted) must be NEGATIVE. Inflows must be POSITIVE.",
    )
    fv: float = Field(
        default=0.0,
        description="Future Value. Accumulated end balance or lump-sum maturity cash flow.",
    )
    pmt: float = Field(
        default=0.0, description="The repeating constant periodic payment value."
    )
    rate: float = Field(
        default=0.0,
        description="The periodic interest rate matching the payment frequency interval, expressed as a clean decimal float (e.g., 0.05).",
    )
    n: float = Field(
        default=0.0,
        description="Total number of compounding/payment periods over the life of the horizon.",
    )
    due: bool = Field(
        default=False,
        description="True if payments happen at the BEGINNING of periods (Annuity Due). False if at the END (Ordinary Annuity).",
    )
    reasoning: str = Field(
        description="A concise actuarial breakdown explaining why these specific parameters were extracted and how rate/period frequencies match."
    )


TVM_SYSTEM_PROMPT = """
You are an expert Actuarial Subagent specialized in Time Value of Money (TVM) math problems
involving loans, investments, and level annuity payment streams.
Your goal is to digest financial questions, isolate the unknown target parameter, and cleanly
extract known parameters.

CRITICAL FINANCIAL CALCULATOR SIGN CONVENTION CONVENTIONS:
1. Outflows (deposits made, cash paid out today, loans handed out) MUST BE NEGATIVE.
2. Inflows (loans received today, cash withdrawals, final maturity payouts) MUST BE POSITIVE.
3. If a student borrows money, PV is POSITIVE (cash inflow). The subsequent regular payments (PMT) will be NEGATIVE (outflows to repay).
4. If a student deposits money into an account, PV is NEGATIVE (outflow). The final maturity amount (FV) will be POSITIVE (inflow).

FREQUENCY MATCHING RULES:
- If payments are monthly, you MUST convert the interest rate to a monthly rate, and 'n' to total months.
- Always pay attention to whether payments happen 'at the beginning' (due = true) or 'at the end' (due = false) of periods.
"""


# ==============================================================================
# STAGE 2B: RATE CONVERSION EXTRACTION (nominal <-> effective <-> force <-> periodic)
# ==============================================================================
class RateConversionParams(BaseModel):
    rate: float = Field(
        description="The known rate as a clean decimal (e.g. 0.08 for 8%), never a raw percentage integer."
    )
    from_type: Literal["nominal", "effective", "force", "periodic"] = Field(
        description="The basis the known rate is already expressed in."
    )
    to_type: Literal["nominal", "effective", "force", "periodic"] = Field(
        description="The basis the user wants the rate converted to."
    )
    m: int = Field(
        default=1,
        description="Compounding frequency per year (e.g. 4 for quarterly, 12 for monthly, 2 for semiannual). Use 1 for annual/effective/force conversions where no sub-annual frequency is mentioned.",
    )
    reasoning: str = Field(
        description="A concise explanation of which rate basis was identified and why, including the compounding frequency chosen."
    )


RATE_SYSTEM_PROMPT = """
You are an expert Actuarial Subagent specialized in interest rate conversions.
Your goal is to identify the known rate, the basis it is expressed in (nominal, effective,
force of interest, or periodic), the basis being requested, and the compounding frequency.

RULES:
- "Nominal annual rate compounded monthly/quarterly/etc." -> from_type = 'nominal', with m set to match the frequency.
- "Effective annual rate" -> from_type or to_type = 'effective'.
- "Force of interest" or "continuously compounded" -> 'force'.
- A bare periodic rate for a single compounding period -> 'periodic'.
- Always express the rate itself as a decimal, never as an integer percentage.
"""


# ==============================================================================
# STAGE 2C: VARIABLE FORCE OF INTEREST EXTRACTION (calculus-based problems)
# ==============================================================================
class ForceOfInterestParams(BaseModel):
    delta_expression: str = Field(
        description=(
            "The force of interest as a function of t, written as a plain math expression "
            "using 't' as the variable, e.g. '0.02 + 0.01*t' or '0.03' for a constant force. "
            "Use only +, -, *, /, **, exp(), ln(), sin(), cos(), sqrt() and numeric literals. "
            "No other function names or Python syntax."
        )
    )
    t1: float = Field(default=0.0, description="Start of the time interval to integrate over.")
    t2: float = Field(description="End of the time interval to integrate over.")
    reasoning: str = Field(
        description="A concise explanation of how delta(t) and the integration bounds were identified from the question."
    )


FORCE_SYSTEM_PROMPT = """
You are an expert Actuarial Subagent specialized in variable force of interest problems.
Your goal is to extract the force of interest function delta(t) as a plain math expression
in terms of t, along with the start and end times of the interval to accumulate/discount over.

RULES:
- Express delta(t) using only +, -, *, /, **, exp(), ln(), and numeric literals - no other syntax.
- If a starting time isn't mentioned, assume t1 = 0.
- t2 is normally the time at which the accumulated or discounted value is being evaluated.
"""


# ==============================================================================
# STAGE 2D: EQUATION OF VALUE EXTRACTION (irregular multi-cash-flow problems)
# ==============================================================================
class EquationOfValueParams(BaseModel):
    cash_flows: List[float] = Field(
        description=(
            "The signed amount of each cash flow, in the same order as `times`. "
            "Outflows (money paid out) MUST BE NEGATIVE. Inflows (money received) MUST BE POSITIVE."
        )
    )
    times: List[float] = Field(
        description="The time in years at which each cash flow occurs, same order and length as `cash_flows`."
    )
    rate: Optional[float] = Field(
        default=None,
        description=(
            "A constant interest rate as a decimal, used when the force of interest is NOT variable. "
            "Required unless `delta_expression` is supplied instead. Leave null if delta_expression is used."
        ),
    )
    rate_type: Literal["nominal", "effective", "force", "periodic"] = Field(
        default="effective",
        description="The basis `rate` is expressed in. Defaults to 'effective' annual if unspecified.",
    )
    m: int = Field(
        default=1,
        description="Compounding frequency per year, only relevant when rate_type is 'nominal' or 'periodic'.",
    )
    delta_expression: Optional[str] = Field(
        default=None,
        description=(
            "If the force of interest is variable (a function of t), the expression in terms of 't', "
            "e.g. '0.02 + 0.01*t'. Leave null if a constant `rate` is used instead."
        ),
    )
    valuation_time: float = Field(
        default=0.0,
        description="The time (in years) all cash flows are being valued/compared at. Defaults to 0 (present value) if unspecified.",
    )
    reasoning: str = Field(
        description="A concise explanation of how each cash flow, its timing, and the valuation basis were identified."
    )


EQUATION_OF_VALUE_SYSTEM_PROMPT = """
You are an expert Actuarial Subagent specialized in the equation of value: setting two or
more cash flows equal at a common point in time to check balance, solve for an unknown
payment, or compare payment streams.

RULES:
1. List every cash flow explicitly, in the same order as its corresponding time.
2. Outflows (loans granted, deposits made, premiums paid) MUST BE NEGATIVE.
   Inflows (loans received, withdrawals, benefit payments received) MUST BE POSITIVE.
3. Use a constant `rate` (with rate_type/m) for a flat interest rate, OR a `delta_expression`
   for a variable force of interest - never both.
4. If the question asks whether a loan or transaction is "balanced" or "fair", set
   valuation_time to 0 and expect the resulting net present value to be at or near zero.
"""


# ==============================================================================
# SAFE EXPRESSION PARSING
# Restricts sympy's parser so a delta(t) string can only ever resolve to a
# mathematical expression in t - never arbitrary Python (no __import__, open,
# exec, etc.), even though the string ultimately originates from user input
# relayed through the model.
# ==============================================================================
_SAFE_LOCAL_FUNCS = {
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "sqrt": sp.sqrt,
}
_SAFE_GLOBAL_DICT = {
    "Float": sp.Float,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
}
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _safe_parse_delta(expr_str: str):
    """Parse a delta(t) string into a SymPy expression with no code-execution surface."""
    t = sp.symbols("t")
    local_dict = {"t": t, **_SAFE_LOCAL_FUNCS}
    expr = parse_expr(
        expr_str,
        local_dict=local_dict,
        global_dict=_SAFE_GLOBAL_DICT,
        transformations=_TRANSFORMATIONS,
    )
    return expr, t


# ==============================================================================
# GEMINI EXTRACTION HELPER
# ==============================================================================
def _extract(client: "genai.Client", system_prompt: str, user_prompt: str, schema) -> dict:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"{system_prompt}\n\nUser Query: {user_prompt}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    return json.loads(response.text)


# ==============================================================================
# FORMATTERS (pure functions of extracted params -> markdown report)
# Kept separate from the Gemini calls so the math + formatting can be tested
# independently of the API.
# ==============================================================================
# Plain-language explanations for why a solved PV/FV/PMT comes out positive
# or negative, written the way a tutor would explain it to a student.
_SIGN_REASON = {
    "pv": {
        True: "you're receiving money today, like loan proceeds, which counts as an inflow",
        False: "you're paying money out today, like a deposit or investment, which counts as an outflow",
    },
    "fv": {
        True: "you'll be receiving money at the end of the term, like an investment payout, which counts as an inflow",
        False: "you'll owe a lump sum at the end of the term, which counts as an outflow",
    },
    "pmt": {
        True: "you're receiving a recurring payment, like an annuity payout, which counts as an inflow",
        False: "you're making a recurring payment, like a loan repayment, which counts as an outflow",
    },
}


def _sign_reason(calc_type: str, result: float) -> str:
    """Explain *why* a solved value is positive or negative, the way a tutor would."""
    is_inflow = result > 0
    reason = _SIGN_REASON.get(calc_type, {}).get(is_inflow)
    if reason is None:
        return ""
    sign_word = "positive" if is_inflow else "negative"
    return f"\n\nYou'll notice this comes out {sign_word} (`{result:,.2f}`) - that's because {reason}."


def _plug_in_tvm_numbers(calc_type: str, pv: float, fv: float, pmt: float, rate: float, n: float, timing_value: str, result: float) -> str:
    """
    Show the formula worked out with actual numbers substituted in, the way
    a tutor would write it out step by step. Returns '' for nper/rate, which
    are solved numerically and don't reduce to one clean substitution line.
    """
    if calc_type not in ("fv", "pv", "pmt"):
        return ""

    growth = (1.0 + rate) ** n if n else 1.0
    a_n = tvm_tools._annuity_factor(rate, n, timing_value)

    if calc_type == "fv":
        s_n = a_n * growth
        return f"FV = -[{pv:,.2f} \u00D7 {growth:.4f} + {pmt:,.2f} \u00D7 {s_n:.4f}] = {result:,.2f}"
    if calc_type == "pv":
        discount = 1.0 / growth if growth else 0.0
        return f"PV = -[{pmt:,.2f} \u00D7 {a_n:.4f} + {fv:,.2f} \u00D7 {discount:.4f}] = {result:,.2f}"
    if calc_type == "pmt":
        discount = 1.0 / growth if growth else 0.0
        return f"PMT = -({pv:,.2f} + {fv:,.2f} \u00D7 {discount:.4f}) / {a_n:.4f} = {result:,.2f}"
    return ""


def format_tvm_annuity(extracted: dict) -> str:
    calc_type = extracted.get("calc_type", "").strip().lower()
    pv = extracted.get("pv", 0.0)
    fv = extracted.get("fv", 0.0)
    pmt = extracted.get("pmt", 0.0)
    rate = extracted.get("rate", 0.0)
    n = extracted.get("n", 0.0)
    due = extracted.get("due", False)
    reasoning = extracted.get("reasoning", "")

    timing_value = "begin" if due else "end"
    timing_label = "beginning of each period (annuity-due)" if due else "end of each period (ordinary annuity)"

    if calc_type == "fv":
        result = tvm_tools.solve_tvm(pv=pv, pmt=pmt, rate=rate, n=n, fv=None, payment_timing=timing_value)
        formula_used = "FV = -[PV \u00D7 (1+r)\u207F + PMT \u00D7 s\u2099]"
        target_label, result_display = "Future Value (FV)", f"${abs(result):,.2f}"
    elif calc_type == "pv":
        result = tvm_tools.solve_tvm(fv=fv, pmt=pmt, rate=rate, n=n, pv=None, payment_timing=timing_value)
        formula_used = "PV = -[PMT \u00D7 a\u2099 + FV \u00D7 (1+r)\u207B\u207F]"
        target_label, result_display = "Present Value (PV)", f"${abs(result):,.2f}"
    elif calc_type == "pmt":
        result = tvm_tools.solve_tvm(pv=pv, fv=fv, rate=rate, n=n, pmt=None, payment_timing=timing_value)
        formula_used = "PMT = -(PV + FV \u00D7 (1+r)\u207B\u207F) / a\u2099"
        target_label, result_display = "Payment (PMT)", f"${abs(result):,.2f}"
    elif calc_type == "nper":
        result = tvm_tools.solve_tvm(pv=pv, fv=fv, pmt=pmt, rate=rate, n=None, payment_timing=timing_value)
        formula_used = "n = ln(ratio) / ln(1+r)"
        target_label, result_display = "Number of Periods (n)", f"{abs(result):,.2f} periods"
    elif calc_type == "rate":
        result = tvm_tools.solve_tvm(pv=pv, fv=fv, pmt=pmt, n=n, rate=None, payment_timing=timing_value)
        formula_used = "solved numerically (Newton-Raphson)"
        target_label, result_display = "Periodic Interest Rate (r)", f"{abs(result) * 100:.4f}%"
    else:
        raise tvm_tools.TVMError(f"Unsupported calculation target mapping pattern: `{calc_type}`.")

    given_lines = []
    if calc_type != "pv":
        given_lines.append(f"- Present Value (PV): ${pv:,.2f}")
    if calc_type != "fv":
        given_lines.append(f"- Future Value (FV): ${fv:,.2f}")
    if calc_type != "pmt":
        given_lines.append(f"- Payment (PMT): ${pmt:,.2f}")
    if calc_type != "rate":
        given_lines.append(f"- Rate per period: {rate * 100:.4f}%")
    if calc_type != "nper":
        given_lines.append(f"- Number of periods (n): {n:.2f}")
    given_lines.append(f"- Payments occur at the {timing_label}")
    given_block = "\n".join(given_lines)

    plugged_in = _plug_in_tvm_numbers(calc_type, pv, fv, pmt, rate, n, timing_value, result)
    plugged_in_block = f"\n\n**Plugging in your numbers:**\n`{plugged_in}`" if plugged_in else ""

    return f"""{reasoning}

**Given:**
{given_block}

**Formula:**
`{formula_used}`{plugged_in_block}

---

### \u2705 Answer
**{target_label} = {result_display}**{_sign_reason(calc_type, result)}"""


def _rate_conversion_steps(rate: float, from_type: str, to_type: str, m: int) -> str:
    """
    convert_rate() always routes through the effective annual rate internally.
    Show that same two-step path with actual numbers, the way a tutor would
    work it out on paper.
    """
    lines = []

    if from_type == "nominal":
        effective = (1.0 + rate / m) ** m - 1.0
        lines.append(f"effective = (1 + {rate:.4f}/{m})^{m} - 1 = {effective:.6f}")
    elif from_type == "force":
        effective = math.exp(rate) - 1.0
        lines.append(f"effective = e^{rate:.4f} - 1 = {effective:.6f}")
    elif from_type == "periodic":
        effective = (1.0 + rate) ** m - 1.0
        lines.append(f"effective = (1 + {rate:.4f})^{m} - 1 = {effective:.6f}")
    else:  # already effective
        effective = rate
        lines.append(f"effective = {rate:.4f} (already an effective annual rate)")

    if to_type == "nominal":
        result = m * ((1.0 + effective) ** (1.0 / m) - 1.0)
        lines.append(f"nominal = {m} \u00D7 [(1 + {effective:.6f})^(1/{m}) - 1] = {result:.6f}")
    elif to_type == "force":
        result = math.log(1.0 + effective)
        lines.append(f"force = ln(1 + {effective:.6f}) = {result:.6f}")
    elif to_type == "periodic":
        result = (1.0 + effective) ** (1.0 / m) - 1.0
        lines.append(f"periodic = (1 + {effective:.6f})^(1/{m}) - 1 = {result:.6f}")
    # if to_type is "effective", the first step already produced the answer

    return "\n".join(lines)


def _article(word: str) -> str:
    """Return 'an' before a vowel sound, 'a' otherwise."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def format_rate_conversion(extracted: dict) -> str:
    rate = extracted.get("rate", 0.0)
    from_type = extracted.get("from_type", "")
    to_type = extracted.get("to_type", "")
    m = extracted.get("m", 1)
    reasoning = extracted.get("reasoning", "")

    result = tvm_tools.convert_rate(rate, from_type=from_type, to_type=to_type, m=m)
    steps = _rate_conversion_steps(rate, from_type, to_type, m)

    return f"""{reasoning}

**Given:**
- Known rate: {rate * 100:.4f}% ({from_type})
- Converting to: {to_type}
- Compounding frequency (m): {m}

**Formula:**
`convert_rate({from_type} \u2192 {to_type}, m={m})`

**Plugging in your numbers:**
{steps}

---

### \u2705 Answer
**{to_type.capitalize()} rate = {result * 100:.4f}%**

In other words, {_article(from_type)} {from_type} rate of {rate * 100:.4f}% is the same thing as {_article(to_type)} {to_type} rate of {result * 100:.4f}% - just expressed on a different compounding basis."""


def format_force_of_interest(extracted: dict) -> str:
    delta_str = extracted.get("delta_expression", "")
    t1 = extracted.get("t1", 0.0)
    t2 = extracted.get("t2", 0.0)
    reasoning = extracted.get("reasoning", "")

    try:
        delta_expr, t_sym = _safe_parse_delta(delta_str)
    except Exception as parse_err:
        raise tvm_tools.TVMError(
            f"Could not parse force of interest expression `{delta_str}`: {parse_err}"
        )

    accumulation_factor = tvm_tools.variable_force_of_interest(delta_expr, t1=t1, t2=t2, variable=t_sym)
    discount_factor = 1.0 / accumulation_factor
    integral_value = float(sp.N(sp.integrate(delta_expr, (t_sym, t1, t2))))

    return f"""{reasoning}

**Given:**
- Force of interest: \u03B4(t) = {delta_str}
- Time interval: t = {t1:.4f} to t = {t2:.4f}

**Formula:**
`A(t1,t2) = exp( \u222B \u03B4(t) dt  from t1 to t2 )`

**Plugging in your numbers:**
`\u222B \u03B4(t) dt from {t1:.4f} to {t2:.4f} = {integral_value:.6f}`
`A = e^{integral_value:.6f} = {accumulation_factor:.6f}`

---

### \u2705 Answer
**Accumulation factor = {accumulation_factor:.6f}**

That means every $1 grows to ${accumulation_factor:.6f} over this interval. Flip it around and the discount factor (v) - what $1 in the future is worth today - works out to {discount_factor:.6f}."""


def _eov_plugged_in(cash_flows: list, times: list, valuation_time: float, rate: float, rate_type: str, m: int, npv: float) -> str:
    """
    Show each cash flow discounted/accumulated with its actual factor, the
    way a tutor would write out a value equation term by term. Skipped when
    there are too many cash flows to stay readable, or when a delta(t)
    function is used instead of a flat rate.
    """
    if len(cash_flows) > 6:
        return ""
    periodic = tvm_tools.convert_rate(rate, from_type=rate_type, to_type="periodic", m=m)
    terms = []
    for cf, t in zip(cash_flows, times):
        exponent = (t - valuation_time) * m
        factor = (1.0 + periodic) ** (-exponent)
        terms.append(f"{cf:,.2f} \u00D7 {factor:.4f}")
    return " + ".join(terms) + f" = {npv:,.2f}"


def _eov_reason(npv: float) -> str:
    """Explain what the balance/imbalance actually means, the way a tutor would."""
    if abs(npv) < 1e-6:
        return (
            "\n\nThis comes out to $0.00, which means the cash flows are perfectly "
            "balanced at this rate - what's paid back is worth exactly what was "
            "borrowed, once interest is accounted for."
        )
    if npv > 0:
        return (
            f"\n\nThis comes out positive (`{npv:,.2f}`), which means the inflows "
            "are worth more than the outflows at this rate - in other words, whoever "
            "receives money is coming out ahead."
        )
    return (
        f"\n\nThis comes out negative (`{npv:,.2f}`), which means the outflows are "
        "worth more than the inflows at this rate - in other words, whoever pays "
        "is giving up more than they're getting back."
    )


def format_equation_of_value(extracted: dict) -> str:
    cash_flows = extracted.get("cash_flows", [])
    times = extracted.get("times", [])
    rate = extracted.get("rate")
    rate_type = extracted.get("rate_type", "effective")
    m = extracted.get("m", 1)
    delta_str = extracted.get("delta_expression")
    valuation_time = extracted.get("valuation_time", 0.0)
    reasoning = extracted.get("reasoning", "")

    if len(cash_flows) != len(times):
        raise tvm_tools.TVMError(
            f"Extracted {len(cash_flows)} cash flows but {len(times)} times - lists must match."
        )

    plugged_in = ""
    if delta_str:
        try:
            delta_expr, t_sym = _safe_parse_delta(delta_str)
        except Exception as parse_err:
            raise tvm_tools.TVMError(
                f"Could not parse force of interest expression `{delta_str}`: {parse_err}"
            )
        npv = tvm_tools.equation_of_value(
            cash_flows, times, delta=delta_expr, valuation_time=valuation_time
        )
        basis_label = f"variable force of interest \u03B4(t) = {delta_str}"
    else:
        if rate is None:
            raise tvm_tools.TVMError(
                "Either a constant `rate` or a `delta_expression` must be provided."
            )
        npv = tvm_tools.equation_of_value(
            cash_flows, times, rate=rate, rate_type=rate_type, m=m, valuation_time=valuation_time
        )
        basis_label = f"a constant {rate_type} rate of {rate * 100:.4f}% (m={m})"
        plugged_in = _eov_plugged_in(cash_flows, times, valuation_time, rate, rate_type, m, npv)

    rows = "\n".join(
        f"| {t:.4f} | ${cf:,.2f} |" for t, cf in zip(times, cash_flows)
    )
    plugged_in_block = f"\n\n**Plugging in your numbers:**\n`{plugged_in}`" if plugged_in else ""

    return f"""{reasoning}

**Given:**
- Valuation basis: {basis_label}
- Valuation time: t = {valuation_time:.4f}

**Cash flow timeline:**

| Time (t) | Cash Flow |
| :--- | :--- |
{rows}

**Formula:**
`NPV = \u03A3 CF_t \u00D7 v(t, valuation_time)`{plugged_in_block}

---

### \u2705 Answer
**Net value at t={valuation_time:.4f} = ${npv:,.4f}**{_eov_reason(npv)}"""


# ==============================================================================
# SINGLE-QUESTION SOLVER
# Classifies one self-contained question and routes it to the matching
# specialist subagent. This is the same logic the orchestrator always ran -
# now factored out so it can be called once per part of a multi-part question.
# ==============================================================================
def _solve_single_question(client: "genai.Client", question: str) -> str:
    try:
        classification = _extract(client, CLASSIFY_SYSTEM_PROMPT, question, ProblemClassification)
        problem_type = classification.get("problem_type")
    except Exception as e:
        return f"\u274C **Classification Error:** {str(e)}"

    try:
        if problem_type == "tvm_annuity":
            extracted = _extract(client, TVM_SYSTEM_PROMPT, question, TVMParameters)
            return format_tvm_annuity(extracted)
        elif problem_type == "rate_conversion":
            extracted = _extract(client, RATE_SYSTEM_PROMPT, question, RateConversionParams)
            return format_rate_conversion(extracted)
        elif problem_type == "force_of_interest":
            extracted = _extract(client, FORCE_SYSTEM_PROMPT, question, ForceOfInterestParams)
            return format_force_of_interest(extracted)
        elif problem_type == "equation_of_value":
            extracted = _extract(
                client, EQUATION_OF_VALUE_SYSTEM_PROMPT, question, EquationOfValueParams
            )
            return format_equation_of_value(extracted)
        else:
            return f"\u274C Subagent Error: unrecognized problem_type `{problem_type}`."
    except tvm_tools.TVMError as math_err:
        return f"\u274C **Actuarial Math Engine Error:** {str(math_err)}"
    except Exception as e:
        return f"\u274C **TVM Subagent Pipeline Error:** {str(e)}"


# ==============================================================================
# TOP-LEVEL ORCHESTRATOR
# ==============================================================================
def process_tvm_request(user_prompt: str) -> str:
    """
    Splits the incoming message into one or more self-contained questions
    (handling multi-part questions like "a. ... b. ... c. ..."), solves each
    one through the classification + specialist-subagent pipeline, and
    returns a combined, formatted markdown report.
    """
    client = genai.Client()

    try:
        split_result = _extract(client, SPLIT_SYSTEM_PROMPT, user_prompt, QuestionSplit)
        sub_questions = split_result.get("sub_questions") or []
    except Exception:
        # If splitting itself fails for any reason, fall back to treating the
        # whole message as a single question rather than losing the request.
        sub_questions = []

    if not sub_questions:
        sub_questions = [user_prompt]

    if len(sub_questions) == 1:
        return _solve_single_question(client, sub_questions[0])

    part_labels = "abcdefghijklmnopqrstuvwxyz"
    sections = []
    for i, sub_question in enumerate(sub_questions):
        label = part_labels[i] if i < len(part_labels) else str(i + 1)
        answer = _solve_single_question(client, sub_question)
        sections.append(f"## Part ({label})\n*{sub_question}*\n\n{answer}")

    return "\n\n---\n\n".join(sections)