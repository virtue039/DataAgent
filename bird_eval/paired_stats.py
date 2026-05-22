"""Pure-function paired statistics for P4.

Two operations:
- mcnemar_with_correction(b, c): McNemar's chi-square test with continuity
  correction for paired binomial discordance.
- newcombe_mover_diff_ci(...): Newcombe MOVER (Method 10) 95% CI for the
  difference between paired binomial proportions.

References:
- Agresti, A. (2002). Categorical Data Analysis, 2nd ed., §10.1.
- Newcombe, R. G. (1998). "Improved confidence intervals for the difference
  between binomial proportions based on paired data." Statistics in Medicine
  17(22), 2635-2650.
"""
from __future__ import annotations

import math

_Z_95 = 1.959963984540054  # 1.96 to 16 digits


def mcnemar_with_correction(b: int, c: int) -> tuple[float, float]:
    """McNemar's chi-square test with Yates continuity correction.

    Inputs:
    - b: number of pairs where A is correct AND B is wrong.
    - c: number of pairs where A is wrong AND B is correct.

    Returns: (chi2_statistic, p_value)

    Edge case: if b + c == 0 (no discordant pairs), returns (0.0, 1.0).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # p-value: 1 - CDF of chi-square with 1 df at `chi2`.
    # Chi-square with 1 df CDF: erf(sqrt(x/2)).
    # So 1-CDF = erfc(sqrt(x/2)).
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


def _wilson_ci(correct: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson 95% CI for a single binomial proportion. Used internally by MOVER."""
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def newcombe_mover_diff_ci(
    a_correct: int,
    a_total: int,
    b_correct: int,
    b_total: int,
    b_count: int,
    c_count: int,
    z: float = _Z_95,
) -> tuple[float, float]:
    """Newcombe MOVER (Method 10) 95% CI for the paired difference p_A - p_B.

    Inputs:
    - a_correct, a_total: marginal counts for A (a_total == b_total for paired).
    - b_correct, b_total: marginal counts for B.
    - b_count: # pairs where A correct AND B wrong.
    - c_count: # pairs where A wrong AND B correct.

    Returns: (lo, hi) 95% confidence bounds on p_A - p_B.

    The MOVER method combines the individual Wilson intervals via a
    correlation correction. See Newcombe (1998) Eq. (10).
    """
    n = a_total
    if n == 0:
        return -1.0, 1.0
    p_a = a_correct / n
    p_b = b_correct / n

    # Individual Wilson intervals.
    l1, u1 = _wilson_ci(a_correct, n, z)
    l2, u2 = _wilson_ci(b_correct, n, z)

    # Correlation correction term phi (Newcombe Eq. 10):
    # phi = (b*c - (a*d if applicable)) / sqrt((b+a)*(b+d)*(c+a)*(c+d))
    # In the standard 2x2 paired table:
    #   a = # both correct
    #   b = # A correct, B wrong (input above)
    #   c = # A wrong, B correct (input above)
    #   d = # both wrong
    a_both = a_correct - b_count  # in A correct ∩ pair → both correct or B wrong;
                                  # a_both = (A correct) ∩ (B correct)
    d_both = n - a_both - b_count - c_count

    # phi: Pearson-style correlation surrogate; if any marginal is 0, set phi=0.
    marg = (a_both + b_count) * (a_both + c_count) * (b_count + d_both) * (c_count + d_both)
    if marg == 0:
        phi = 0.0
    else:
        phi = (a_both * d_both - b_count * c_count) / math.sqrt(marg)

    # MOVER lower bound:
    # lo = p_a - p_b - sqrt((p_a - l1)^2 - 2*phi*(p_a - l1)*(u2 - p_b) + (u2 - p_b)^2)
    diff = p_a - p_b
    da_lo = p_a - l1
    db_hi = u2 - p_b
    lo_radical = max(0.0, da_lo * da_lo - 2.0 * phi * da_lo * db_hi + db_hi * db_hi)
    lo = diff - math.sqrt(lo_radical)

    # MOVER upper bound:
    da_hi = u1 - p_a
    db_lo = p_b - l2
    hi_radical = max(0.0, da_hi * da_hi - 2.0 * phi * da_hi * db_lo + db_lo * db_lo)
    hi = diff + math.sqrt(hi_radical)

    return lo, hi


def format_paired_table(rows: list[dict], n: int | None = None) -> str:
    """Render a list of paired-comparison rows as a stdout table.

    Each row: {pair_name, a_pct, b_pct, diff_pct, chi2, p_value, lo, hi}
    `n` is the per-pair sample size (printed in the header). If None,
    the header omits the count.
    """
    n_str = f"n={n}" if n is not None else "paired"
    lines = [
        f"==== P4 SECTION 6: Paired comparisons ({n_str}) ====",
        "",
        f"{'A vs B':<22} | {'A%':>6} | {'B%':>6} | {'diff':>7} | "
        f"{'χ²':>8} | {'p-value':>10}",
        "-" * 80,
    ]
    for r in rows:
        p_str = f"{r['p_value']:.3e}" if r["p_value"] < 1e-3 else f"{r['p_value']:.3f}"
        lines.append(
            f"{r['pair_name']:<22} | {r['a_pct']:>6.2f} | {r['b_pct']:>6.2f} | "
            f"{r['diff_pct']:>+7.2f} | {r['chi2']:>8.2f} | {p_str:>10}"
        )
    lines.append("")
    lines.append("Newcombe MOVER 95% CI for paired difference:")
    for r in rows:
        verdict = "overlaps 0" if r["lo"] <= 0 <= r["hi"] else "strictly different"
        lines.append(f"  {r['pair_name']:<22} = [{r['lo']*100:>+6.2f}, "
                     f"{r['hi']*100:>+6.2f}] pp   ({verdict})")
    return "\n".join(lines)
