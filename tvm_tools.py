"""
Pure mathematical Time Value of Money (TVM) backend logic.

Sign convention: outflows are negative, inflows are positive.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Sequence, Union

import sympy as sp
from sympy import Expr, Symbol, exp, integrate, ln, symbols

Number = Union[int, float]
RateType = Literal["nominal", "effective", "force", "periodic"]
PaymentTiming = Literal["end", "begin"]

import pandas as pd

class TVMError(ValueError):
    """Raised when TVM inputs are invalid or a quantity cannot be solved."""


def _as_float(value: Union[Number, Expr]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(sp.N(value))


def enforce_cash_flow_sign(amount: Number, direction: Literal["inflow", "outflow"]) -> float:
    """
    Return a signed cash flow amount.

    Inflows are positive; outflows are negative.
    """
    magnitude = abs(float(amount))
    if direction == "inflow":
        return magnitude
    if direction == "outflow":
        return -magnitude
    raise TVMError(f"direction must be 'inflow' or 'outflow', got {direction!r}")


def _validate_signed_cash_flows(cash_flows: Sequence[Number]) -> list[float]:
    return [float(cf) for cf in cash_flows]


def _annuity_factor(rate: float, n: float, timing: PaymentTiming) -> float:
    if n == 0:
        return 0.0
    if abs(rate) < 1e-15:
        factor = n
    else:
        factor = (1.0 - (1.0 + rate) ** (-n)) / rate
    if timing == "begin":
        factor *= 1.0 + rate
    return factor


def _solve_rate_newton(
    pv: float,
    fv: float,
    pmt: float,
    n: float,
    timing: PaymentTiming,
    guess: float = 0.05,
) -> float:
    """Solve for the periodic interest rate with Newton-Raphson."""

    def f(rate: float) -> float:
        if abs(rate) < 1e-15:
            return pv + pmt * n + fv
        growth = (1.0 + rate) ** n
        annuity = _annuity_factor(rate, n, timing)
        return pv + pmt * annuity + fv * (1.0 + rate) ** (-n)

    def df(rate: float) -> float:
        if abs(rate) < 1e-15:
            return 0.0
        v_n = (1.0 + rate) ** (-n)
        if timing == "end":
            d_annuity = (n * v_n) / rate - (1.0 - v_n) / (rate**2)
        else:
            ordinary = (1.0 - v_n) / rate
            d_annuity = (1.0 + rate) * (
                (n * v_n) / rate - (1.0 - v_n) / (rate**2)
            ) + ordinary
        d_fv = -n * fv * (1.0 + rate) ** (-n - 1)
        return pmt * d_annuity + d_fv

    rate = guess
    for _ in range(100):
        value = f(rate)
        if abs(value) < 1e-12:
            return rate
        derivative = df(rate)
        if abs(derivative) < 1e-15:
            break
        rate -= value / derivative
        if rate <= -1.0:
            rate = (rate + 1.0) / 2.0
    raise TVMError("Failed to solve for interest rate; try a different guess.")


def solve_tvm(
    *,
    pv: Number | None = None,
    fv: Number | None = None,
    pmt: Number | None = None,
    n: Number | None = None,
    rate: Number | None = None,
    payment_timing: PaymentTiming = "end",
    rate_guess: float = 0.05,
) -> float:
    """
    Solve for the single unknown among present value, future value, payment,
    number of periods, or periodic interest rate.

    Cash flows follow the sign convention (outflows negative, inflows positive).
    End-of-period payments are the default; use payment_timing='begin' for annuity due.
    """
    unknowns = [name for name, value in (
        ("pv", pv), ("fv", fv), ("pmt", pmt), ("n", n), ("rate", rate)
    ) if value is None]
    if len(unknowns) != 1:
        raise TVMError("Exactly one of pv, fv, pmt, n, or rate must be None.")

    known = {
        "pv": 0.0 if pv is None else float(pv),
        "fv": 0.0 if fv is None else float(fv),
        "pmt": 0.0 if pmt is None else float(pmt),
        "n": 0.0 if n is None else float(n),
        "rate": 0.0 if rate is None else float(rate),
    }
    target = unknowns[0]

    if target != "n" and known["n"] < 0:
        raise TVMError("Number of periods must be non-negative.")
    if target != "rate" and known["rate"] <= -1.0:
        raise TVMError("Periodic rate must be greater than -100%.")

    if target == "rate":
        return _solve_rate_newton(
            known["pv"], known["fv"], known["pmt"], known["n"], payment_timing, rate_guess
        )

    i = known["rate"]
    periods = known["n"]
    annuity = _annuity_factor(i, periods, payment_timing)

    if target == "pv":
        if abs(i) < 1e-15:
            return -(known["fv"] + known["pmt"] * periods)
        return -(known["pmt"] * annuity + known["fv"] * (1.0 + i) ** (-periods))

    if target == "fv":
        if abs(i) < 1e-15:
            return -(known["pv"] + known["pmt"] * periods)
        return -(
            known["pv"] * (1.0 + i) ** periods
            + known["pmt"] * (((1.0 + i) ** periods - 1.0) / i)
            * (1.0 + i if payment_timing == "begin" else 1.0)
        )

    if target == "pmt":
        if abs(annuity) < 1e-15:
            raise TVMError("Payment cannot be solved when the annuity factor is zero.")
        if abs(i) < 1e-15:
            return -(known["pv"] + known["fv"]) / periods
        return -(known["pv"] + known["fv"] * (1.0 + i) ** (-periods)) / annuity

    if target == "n":
        i = known["rate"]  
        
        if known["pmt"] == 0.0:
            if known["pv"] == 0.0 or known["fv"] == 0.0:
                raise TVMError("Cannot solve n when pv, fv, and pmt are all zero.")
            if known["pv"] * known["fv"] >= 0:
                raise TVMError("pv and fv must have opposite signs to solve for n with pmt=0.")
            ratio = -known["fv"] / known["pv"]
            if ratio <= 0:
                raise TVMError("Invalid pv/fv ratio for solving n.")
            if abs(i) < 1e-15: 
                raise TVMError("Cannot solve n from pv and fv when rate is zero.")
            return math.log(ratio) / math.log(1.0 + i)
        
        if abs(i) < 1e-15:
            if known["pv"] + known["fv"] + known["pmt"] * 0 != 0:
                return -known["pv"] / known["pmt"] if known["pmt"] != 0 else 0.0
            raise TVMError("Cannot solve n with zero rate unless flows already balance at n=0.")

        def balance(periods: float) -> float:
            return (
                known["pv"]
                + known["pmt"] * _annuity_factor(i, periods, payment_timing)
                + known["fv"] * (1.0 + i) ** (-periods)
            )

        lo, hi = 0.0, 1.0
        while balance(hi) * balance(lo) > 0 and hi < 1e6:
            hi *= 2.0
        if balance(hi) * balance(lo) > 0:
            raise TVMError("Could not bracket a root for n.")
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if balance(mid) == 0 or (hi - lo) < 1e-12:
                return mid
            if balance(lo) * balance(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    raise TVMError(f"Unsupported unknown: {target}")


def convert_rate(
    rate: Number,
    *,
    from_type: RateType,
    to_type: RateType,
    m: int = 1,
) -> float:
    """
    Convert among nominal, effective, force-of-interest, and periodic rates.

    Parameters
    ----------
    rate:
        Rate expressed in ``from_type`` units (decimal, not percent).
    from_type / to_type:
        One of ``nominal``, ``effective``, ``force``, or ``periodic``.
    m:
        Compounding frequency per year for nominal/periodic conversions.
    """
    if m <= 0:
        raise TVMError("Compounding frequency m must be positive.")

    r = float(rate)

    if from_type == "nominal":
        effective = (1.0 + r / m) ** m - 1.0
    elif from_type == "effective":
        effective = r
    elif from_type == "force":
        effective = math.exp(r) - 1.0
    elif from_type == "periodic":
        effective = (1.0 + r) ** m - 1.0
    else:
        raise TVMError(f"Unknown from_type: {from_type!r}")

    if to_type == "effective":
        return effective
    if to_type == "nominal":
        return m * ((1.0 + effective) ** (1.0 / m) - 1.0)
    if to_type == "force":
        return math.log(1.0 + effective)
    if to_type == "periodic":
        return (1.0 + effective) ** (1.0 / m) - 1.0
    raise TVMError(f"Unknown to_type: {to_type!r}")


def variable_force_of_interest(
    delta: Union[Expr, Callable[[Number], Number], Number],
    t1: Number = 0,
    t2: Number | None = None,
    *,
    variable: Symbol | None = None,
) -> float:
    """
    Compute the accumulation factor exp(integral of delta over [t1, t2]).

    ``delta`` may be a constant, a callable delta(t), or a SymPy expression in
    ``variable`` (default symbol ``t``).
    """
    start = float(t1)
    end = float(t2 if t2 is not None else t1)

    if isinstance(delta, (int, float)):
        return math.exp(float(delta) * (end - start))

    if callable(delta):
        if end == start:
            return 1.0
        steps = max(1000, int(abs(end - start) * 1000))
        step = (end - start) / steps
        total = 0.0
        for k in range(steps):
            left = start + k * step
            mid = left + step / 2.0
            total += float(delta(mid)) * step
        return math.exp(total)

    t = variable if variable is not None else symbols("t")
    integral_value = integrate(delta, (t, start, end))
    return _as_float(exp(integral_value))


def discount_factor_variable_force(
    delta: Union[Expr, Callable[[Number], Number], Number],
    t1: Number = 0,
    t2: Number = 0,
    *,
    variable: Symbol | None = None,
) -> float:
    """Present-value discount factor v(t1, t2) = exp(-integral of delta over [t1, t2])."""
    return 1.0 / variable_force_of_interest(delta, t1, t2, variable=variable)


def equation_of_value(
    cash_flows: Sequence[Number],
    times: Sequence[Number],
    rate: Number | None = None,
    *,
    delta: Union[Expr, Callable[[Number], Number], Number, None] = None,
    m: int = 1,
    rate_type: RateType = "effective",
    valuation_time: Number = 0,
) -> float:
    """
    Evaluate the equation of value: sum of signed cash flows brought to ``valuation_time``.

    Returns zero when the cash flows are in actuarial balance. Uses a constant
    effective rate or a variable force of interest supplied via ``delta``.
    """
    flows = _validate_signed_cash_flows(cash_flows)
    time_list = [float(t) for t in times]
    if len(flows) != len(time_list):
        raise TVMError("cash_flows and times must have the same length.")

    total = 0.0
    v_time = float(valuation_time)

    for amount, cash_time in zip(flows, time_list):
        if amount == 0.0:
            continue
        if delta is not None:
            if cash_time >= v_time:
                factor = discount_factor_variable_force(delta, v_time, cash_time)
            else:
                factor = 1.0 / variable_force_of_interest(delta, cash_time, v_time)
        else:
            if rate is None:
                raise TVMError("Either rate or delta must be provided.")
            periodic = convert_rate(rate, from_type=rate_type, to_type="periodic", m=m)
            exponent = (cash_time - v_time) * m
            factor = (1.0 + periodic) ** (-exponent)
        total += amount * factor

    return total


def fisher_equation(
    *,
    nominal: Number | None = None,
    real: Number | None = None,
    inflation: Number | None = None,
) -> float:
    """
    Solve the Fisher equation: (1 + nominal) = (1 + real)(1 + inflation).

    Exactly one argument must be None.
    """
    unknowns = [
        name
        for name, value in (("nominal", nominal), ("real", real), ("inflation", inflation))
        if value is None
    ]
    if len(unknowns) != 1:
        raise TVMError("Exactly one of nominal, real, or inflation must be None.")

    i = None if nominal is None else float(nominal)
    r = None if real is None else float(real)
    pi = None if inflation is None else float(inflation)

    if unknowns[0] == "nominal":
        return (1.0 + r) * (1.0 + pi) - 1.0
    if unknowns[0] == "real":
        return (1.0 + i) / (1.0 + pi) - 1.0
    return (1.0 + i) / (1.0 + r) - 1.0


def _assert_close(actual: float, expected: float, tol: float = 1e-9, label: str = "") -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def run_golden_set() -> None:
    """Run ten distinct TVM checks with hand-verified expected values."""
    tests_passed = 0

    # 1. Future value of a single outflow (lump-sum investment)
    fv = solve_tvm(pv=-1000.0, rate=0.05, n=10, pmt=0.0)
    _assert_close(fv, 1628.89462677744, label="solve_tvm fv lump sum")
    tests_passed += 1

    # 2. Present value of a future inflow
    pv = solve_tvm(fv=1000.0, rate=0.04, n=5, pmt=0.0)
    _assert_close(pv, -821.92710664479, label="solve_tvm pv discount")
    tests_passed += 1

    # 3. Level payment (ordinary annuity) that amortizes a loan
    pmt = solve_tvm(pv=10000.0, fv=0.0, rate=0.06 / 12, n=36)
    _assert_close(pmt, -304.21937451555704, label="solve_tvm loan payment")
    tests_passed += 1

    # 4. Solve periodic rate for a two-cash-flow example
    rate = solve_tvm(pv=-1000.0, fv=1500.0, n=5, pmt=0.0)
    _assert_close(rate, 0.08447177119769853, tol=1e-12, label="solve_tvm rate")
    tests_passed += 1

    # 5. Nominal annual to effective annual (6% compounded quarterly)
    effective = convert_rate(0.06, from_type="nominal", to_type="effective", m=4)
    _assert_close(effective, 0.061363550, tol=1e-9, label="convert_rate nominal->effective")
    tests_passed += 1

    # 6. Effective rate to force of interest
    force = convert_rate(0.05, from_type="effective", to_type="force")
    _assert_close(force, math.log(1.05), label="convert_rate effective->force")
    tests_passed += 1

    # 7. Constant force of interest accumulation via SymPy integration
    t = symbols("t")
    acc = variable_force_of_interest(ln(1.05), t1=0, t2=10, variable=t)
    _assert_close(acc, 1.05**10, label="variable_force constant")
    tests_passed += 1

    # 8. Linear force of interest delta(t) = 0.02 + 0.01 t over [0, 2]
    delta_linear = 0.02 + 0.01 * t
    acc_linear = variable_force_of_interest(delta_linear, t1=0, t2=2, variable=t)
    _assert_close(acc_linear, math.exp(0.06), label="variable_force linear")
    tests_passed += 1

    # 9. Equation of value for a balanced loan (net present value equals zero)
    loan_pv = 10000.0
    monthly_rate = 0.06 / 12
    payment = 304.21937451555704
    flows = [loan_pv] + [enforce_cash_flow_sign(payment, "outflow")] * 36
    times = [0.0] + [k / 12 for k in range(1, 37)]
    npv = equation_of_value(
        flows,
        times,
        rate=monthly_rate,
        rate_type="periodic",
        m=12,
    )
    _assert_close(npv, 0.0, tol=1e-6, label="equation_of_value loan")
    tests_passed += 1

    # 10. Fisher equation: real rate from nominal and inflation
    real_rate = fisher_equation(nominal=0.08, inflation=0.03)
    _assert_close(real_rate, 0.048543689, tol=1e-9, label="fisher_equation real")
    tests_passed += 1

    print(f"run_golden_set: all {tests_passed} checks passed.")

def generate_amortization_schedule(*, pv: float, annual_rate: float, periods_per_year: int, years: int) -> dict:
    """
    Computes a full amortization schedule using discrete periodic compounding.
    Implements the lean_summary() principle for the LLM return payload.
    """
    # Enforce input format guard (e.g., auto-correct rate 6 or 6.0 into 0.06)
    if annual_rate >= 1.0:
        annual_rate = annual_rate / 100.0

    r = annual_rate / periods_per_year
    n = int(years * periods_per_year)
    
    # Calculate the fixed regular payment (PMT)
    pmt = (pv * r) / (1 - (1 + r)**(-n))
    
    balance = pv
    schedule_data = []
    
    for t in range(1, n + 1):
        interest_t = balance * r
        principal_t = pmt - interest_t
        ending_balance = balance - principal_t
        
        # Guard rails for final period rounding errors
        if t == n or ending_balance < 0:
            principal_t = balance
            ending_balance = 0.0
            
        schedule_data.append({
            "Period": t,
            "Beginning_Balance": round(balance, 2),
            "Payment": round(pmt, 2),
            "Interest_Paid": round(interest_t, 2),
            "Principal_Paid": round(principal_t, 2),
            "Ending_Balance": round(ending_balance, 2)
        })
        balance = ending_balance

    df = pd.DataFrame(schedule_data)
    
    # --- 🛡️ LEAN SUMMARY GUARDRAIL ---
    # We pass only vital diagnostic calculation rows back to the LLM context window
    lean_payload = {
        "meta": {
            "total_periods": n,
            "fixed_payment": round(pmt, 2),
            "total_interest": round(df["Interest_Paid"].sum(), 2),
            "total_paid": round(df["Payment"].sum(), 2)
        },
        "first_period": schedule_data[0],
        "mid_period": schedule_data[n // 2],
        "last_period": schedule_data[-1]
    }
    
    return {"full_df": df, "llm_summary": lean_payload}

if __name__ == "__main__":
    run_golden_set()
