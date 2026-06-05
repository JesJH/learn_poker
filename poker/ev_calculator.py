"""
Expected Value calculator at each poker decision node.

EV is the fundamental concept in both poker and quantitative trading:
the probability-weighted average of all possible outcomes.

  EV = Σ (probability_i × outcome_i)

In trading this is the expected P&L of a position:
  EV(trade) = p(up) × gain - p(down) × loss

At a poker decision node we compute EV for three actions:

  EV(fold)  = 0
    You exit the pot. No future gain or loss. Chips already committed
    are a sunk cost (lost regardless). This is why you should never
    "call just because you've already put chips in" — that's the
    sunk cost fallacy.

  EV(call)  = pot × equity - to_call × (1 - equity)
    You pay to_call to see the hand through. You win the pot with
    probability = equity, lose to_call with probability = 1 - equity.

  EV(raise) = fold_equity × pot
              + (1 - fold_equity) × [(pot + raise_amount) × equity - raise_amount]
    Two ways to win: opponents fold immediately (you take pot), or they
    call and you still win at showdown. fold_equity is estimated from
    raise size and opponent tendencies.

Reference:
  Sklansky, D. (1999). The Theory of Poker. Two Plus Two Publishing.
  Chen, B. & Ankenman, J. (2006). The Mathematics of Poker. ConJelCo.
"""
from __future__ import annotations


def compute(
    equity: float,
    pot: int,
    to_call: int,
    raise_amount: int | None = None,
    fold_equity: float = 0.30,
) -> dict:
    """
    Compute EV for all three actions at a decision node.

    Args:
        equity       : win probability from Monte Carlo (0-1)
        pot          : current pot before action
        to_call      : chips to call (0 if checking)
        raise_amount : total raise amount above current bet (None = not shown)
        fold_equity  : assumed probability opponents fold to a raise (default 30%)

    Returns dict with EV for each action and a recommended action.
    """
    # EV of folding is always 0 (relative to current state)
    ev_fold = 0.0

    # EV of calling
    if to_call > 0:
        ev_call = pot * equity - to_call * (1 - equity)
    else:
        # Checking — EV is pot × equity (free to see next card)
        ev_call = pot * equity

    # EV of raising
    if raise_amount and raise_amount > 0:
        ev_raise = (
            fold_equity * pot
            + (1 - fold_equity) * ((pot + raise_amount) * equity - raise_amount)
        )
    else:
        ev_raise = None

    # Best action
    candidates = {"Fold": ev_fold, "Call" if to_call > 0 else "Check": ev_call}
    if ev_raise is not None:
        candidates["Raise"] = ev_raise
    best_action = max(candidates, key=lambda k: candidates[k])

    return {
        "ev_fold": round(ev_fold, 2),
        "ev_call": round(ev_call, 2),
        "ev_check": round(ev_call, 2) if to_call == 0 else None,
        "ev_raise": round(ev_raise, 2) if ev_raise is not None else None,
        "best_action": best_action,
        "fold_equity_assumed": fold_equity,
        "equity_used": equity,
    }


def ev_formula_str() -> str:
    return r"EV = \sum_{i} p_i \cdot x_i"


def call_ev_formula_str() -> str:
    return r"EV(\text{call}) = \text{pot} \times p - \text{to\_call} \times (1 - p)"


def raise_ev_formula_str() -> str:
    return (
        r"EV(\text{raise}) = f_{\text{fold}} \cdot \text{pot} "
        r"+ (1 - f_{\text{fold}}) \cdot [(\text{pot} + \text{raise}) \times p - \text{raise}]"
    )
