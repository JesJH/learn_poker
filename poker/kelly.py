"""
Kelly Criterion bet sizing for poker.

The Kelly Criterion answers: "What fraction of my bankroll should I wager
to maximize long-run logarithmic wealth growth?"

Formula:  f* = (b*p - q) / b
  where:
    f* = fraction of bankroll to bet
    b  = net odds received (pot / cost-to-call)
    p  = probability of winning (equity)
    q  = probability of losing (1 - p)

In poker:
  - "bankroll" = your chip stack
  - "b"        = pot odds ratio (how much you win per dollar risked)
  - "p"        = Monte Carlo equity

Trading analogy:
  Kelly is used by quantitative funds to size positions across uncorrelated
  trading signals. Betting more than Kelly leads to ruin (gambler's ruin);
  betting less sacrifices compounding growth. Half-Kelly is standard in
  practice because Kelly assumes exact knowledge of p, which is never true.

Reference:
  Kelly, J.L. (1956). "A New Interpretation of Information Rate."
  Bell System Technical Journal, 35(4), 917–926.
"""
from __future__ import annotations


def kelly_fraction(equity: float, pot: int, to_call: int) -> dict:
    """
    Compute Kelly optimal bet fraction and recommended bet size.

    Args:
        equity   : win probability (0-1), from Monte Carlo
        pot      : current pot size before the call
        to_call  : chips required to call

    Returns dict with:
        full_kelly_fraction : float  — theoretical optimum
        half_kelly_fraction : float  — practical recommendation
        kelly_bet           : int    — full Kelly bet in chips (based on stack fraction × pot)
        half_kelly_bet      : int    — half-Kelly bet
        positive_ev         : bool   — whether calling/betting has positive EV at all
        explanation         : str
    """
    if to_call <= 0:
        # No call required — player is first to act or checking
        return {
            "full_kelly_fraction": None,
            "half_kelly_fraction": None,
            "kelly_bet": None,
            "half_kelly_bet": None,
            "positive_ev": equity > 0.5,
            "explanation": "No bet to call. Kelly applies when sizing your own bet.",
        }

    # b = net odds: how much you win per dollar risked
    b = pot / to_call
    p = equity
    q = 1 - equity

    f_star = (b * p - q) / b  # Kelly fraction

    if f_star <= 0:
        return {
            "full_kelly_fraction": 0.0,
            "half_kelly_fraction": 0.0,
            "kelly_bet": 0,
            "half_kelly_bet": 0,
            "positive_ev": False,
            "explanation": (
                f"Kelly says don't bet (f* = {round(f_star, 3)}). "
                f"Your equity ({round(p*100, 1)}%) doesn't justify the cost. "
                f"Folding preserves bankroll."
            ),
        }

    half_f = f_star / 2

    # Express as fraction of the pot (useful for bet sizing)
    kelly_bet = int(pot * f_star)
    half_kelly_bet = int(pot * half_f)

    explanation = (
        f"Full Kelly: bet {round(f_star * 100, 1)}% of the pot (${kelly_bet}). "
        f"Half-Kelly (recommended): ${half_kelly_bet}. "
        f"Using half-Kelly accounts for uncertainty in your equity estimate — "
        f"overbetting Kelly risks ruin even with an edge."
    )

    return {
        "full_kelly_fraction": round(f_star, 4),
        "half_kelly_fraction": round(half_f, 4),
        "kelly_bet": kelly_bet,
        "half_kelly_bet": half_kelly_bet,
        "positive_ev": True,
        "explanation": explanation,
    }


def kelly_formula_str() -> str:
    return r"f^* = \frac{b \cdot p - q}{b}"
