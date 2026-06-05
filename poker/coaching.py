"""
Rule-based coaching engine.

Provides two things:
  1. pre_action_tip()  — guidance shown BEFORE the player makes a decision
  2. evaluate_action() — grades the player's choice after they make it
  3. hand_review()     — post-hand list of moments the player could improve
"""
from __future__ import annotations
from .card import Card
from .player import Action
from .hand_evaluator import best_hand, hand_strength, HAND_NAMES, HIGH_CARD, ONE_PAIR, TWO_PAIR


# ---------------------------------------------------------------------------
# Pre-flop hand quality (simplified Chen formula-inspired scoring)
# ---------------------------------------------------------------------------

# Premium starting hands that should almost always be played
PREMIUM_HANDS = {
    frozenset(["A", "A"]), frozenset(["K", "K"]), frozenset(["Q", "Q"]),
    frozenset(["J", "J"]), frozenset(["A", "K"]),
}

STRONG_HANDS = {
    frozenset(["10", "10"]), frozenset(["9", "9"]), frozenset(["A", "Q"]),
    frozenset(["A", "J"]), frozenset(["K", "Q"]),
}


def _preflop_category(hole: list[Card]) -> str:
    ranks = frozenset(c.rank for c in hole)
    suited = hole[0].suit == hole[1].suit
    connected = abs(hole[0].value - hole[1].value) == 1

    if ranks in PREMIUM_HANDS:
        return "premium"
    if ranks in STRONG_HANDS:
        return "strong"
    paired = hole[0].rank == hole[1].rank
    if paired:
        return "medium_pair"
    vals = sorted([c.value for c in hole], reverse=True)
    if vals[0] >= 10 and suited:
        return "suited_broadway"
    if suited and connected:
        return "suited_connector"
    if vals[0] >= 10:
        return "broadway"
    if connected:
        return "connector"
    return "weak"


def _pot_odds(to_call: int, pot: int) -> float | None:
    """Return pot odds as a fraction (call / total pot after call). None if no call needed."""
    if to_call <= 0:
        return None
    return to_call / (pot + to_call)


def _equity_estimate(hole: list[Card], community: list[Card]) -> float:
    """Rough equity: 0-1 estimate of winning likelihood based on hand rank."""
    if not community:
        # Use pre-flop category
        cat = _preflop_category(hole)
        return {
            "premium": 0.72, "strong": 0.60, "medium_pair": 0.52,
            "suited_broadway": 0.50, "suited_connector": 0.48,
            "broadway": 0.46, "connector": 0.44, "weak": 0.35,
        }[cat]
    return hand_strength(hole, community)


# ---------------------------------------------------------------------------
# Position labels
# ---------------------------------------------------------------------------

def position_label(player_index: int, dealer_index: int, num_players: int) -> str:
    offset = (player_index - dealer_index) % num_players
    if num_players <= 2:
        return "Button / Small Blind" if offset == 0 else "Big Blind"
    labels = {1: "Small Blind", 2: "Big Blind"}
    late_start = num_players - 2
    if offset == 0:
        return "Button (best position)"
    if offset in labels:
        return labels[offset]
    if offset >= late_start:
        return "Late Position (good)"
    if offset <= num_players // 2:
        return "Early Position (careful)"
    return "Middle Position"


# ---------------------------------------------------------------------------
# 1. Pre-action tip
# ---------------------------------------------------------------------------

def pre_action_tip(
    hole: list[Card],
    community: list[Card],
    to_call: int,
    pot: int,
    player_index: int,
    dealer_index: int,
    num_players: int,
) -> dict:
    """
    Returns a dict with coaching info shown before the player acts:
      hand_quality  : "Premium" / "Strong" / "Playable" / "Weak"
      hand_name     : e.g. "Two Pair" (empty pre-flop)
      position      : position label
      pot_odds_pct  : float or None
      equity_pct    : float
      advice        : main coaching sentence
      tip_level     : "info" / "warning" / "success"
    """
    equity = _equity_estimate(hole, community)
    odds = _pot_odds(to_call, pot)
    pos = position_label(player_index, dealer_index, num_players)

    if community:
        score, _, hand_name = best_hand(hole, community)
        hand_rank = score[0]
    else:
        hand_name = ""
        hand_rank = -1

    # Pre-flop quality
    cat = _preflop_category(hole)
    quality_map = {
        "premium": "Premium", "strong": "Strong", "medium_pair": "Playable",
        "suited_broadway": "Playable", "suited_connector": "Playable",
        "broadway": "Playable", "connector": "Marginal", "weak": "Weak",
    }
    quality = quality_map[cat]

    # Build advice
    lines = []

    # Hand quality line
    if not community:
        hand_tips = {
            "premium":          "This is one of the best starting hands. Raise to build the pot.",
            "strong":           "Strong starting hand. Raise or call comfortably.",
            "medium_pair":      "A medium pair — playable, but watch for overcards on the board.",
            "suited_broadway":  "High cards of the same suit — good potential. Worth playing.",
            "suited_connector": "Suited connectors can make straights and flushes. Playable in position.",
            "broadway":         "Two high cards, but not suited. Playable, but proceed carefully.",
            "connector":        "Low connectors need cheap flops to be profitable. Consider folding to raises.",
            "weak":             "This hand has weak potential. Folding is usually correct unless you can check for free.",
        }
        lines.append(hand_tips[cat])
    else:
        if hand_rank >= TWO_PAIR:
            lines.append(f"You have {hand_name} — a strong made hand. Consider betting or raising.")
        elif hand_rank == ONE_PAIR:
            lines.append(f"You have {hand_name}. Decent, but vulnerable to two-pair and better hands.")
        else:
            lines.append(f"You have {hand_name} (High Card). You'll need to bluff or improve to win.")

    # Pot odds line (only when a call is required)
    if odds is not None:
        odds_pct = round(odds * 100)
        equity_pct = round(equity * 100)
        if equity_pct > odds_pct + 10:
            lines.append(
                f"Pot odds: you'd pay {odds_pct}% of the pot to call, and your hand wins roughly "
                f"{equity_pct}% of the time — calling has positive expected value."
            )
        elif equity_pct < odds_pct - 5:
            lines.append(
                f"Pot odds: calling costs {odds_pct}% of the pot, but your equity is only ~{equity_pct}%. "
                f"Calling here loses money long-term — folding or raising (as a bluff) may be better."
            )
        else:
            lines.append(
                f"Pot odds: about break-even (call = {odds_pct}%, equity ≈ {equity_pct}%). "
                f"Position and opponent tendencies should guide your decision."
            )

    # Position line
    if "Button" in pos or "Late" in pos:
        lines.append(f"You're in {pos} — you act last, giving you an information advantage.")
    elif "Early" in pos or "Big Blind" in pos or "Small Blind" in pos:
        lines.append(f"You're in {pos} — acting early means others haven't shown their intentions yet. Play tighter.")

    # Tip level
    if quality == "Premium" or (community and hand_rank >= TWO_PAIR):
        tip_level = "success"
    elif quality == "Weak" or (odds is not None and equity * 100 < _pot_odds(to_call, pot) * 100 - 5):
        tip_level = "warning"
    else:
        tip_level = "info"

    # Split advice into labelled categories
    hand_line = lines[0] if len(lines) > 0 else ""
    pot_odds_line = lines[1] if odds is not None and len(lines) > 1 else None
    position_line = lines[-1] if len(lines) > 1 else None
    # Recommendation = distilled action sentence
    if quality in ("Premium", "Strong") or (community and hand_rank >= TWO_PAIR):
        recommendation = "Bet or raise to build the pot — you have a strong hand."
    elif quality == "Weak" and (odds is None or equity * 100 < (odds or 0) * 100):
        recommendation = "Consider folding unless you can act for free."
    elif odds is not None and equity * 100 > (odds * 100 + 10):
        recommendation = "Calling has positive expected value — the pot odds support it."
    elif odds is not None and equity * 100 < (odds * 100 - 5):
        recommendation = "The pot odds don't support calling — fold or bluff-raise instead."
    else:
        recommendation = "Proceed carefully — this is a marginal spot."

    return {
        "hand_quality": quality,
        "hand_name": hand_name,
        "position": pos,
        "pot_odds_pct": round(odds * 100) if odds is not None else None,
        "equity_pct": round(equity * 100),
        "tip_level": tip_level,
        # Structured categories
        "your_hand": hand_line,
        "pot_odds": pot_odds_line,
        "position_advice": position_line,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 2. Post-action evaluation
# ---------------------------------------------------------------------------

def evaluate_action(
    action: Action,
    hole: list[Card],
    community: list[Card],
    to_call: int,
    pot: int,
) -> dict:
    """
    Grade the player's action immediately after they make it.
    Returns: { "grade": "good"/"ok"/"mistake", "explanation": str }
    """
    equity = _equity_estimate(hole, community)
    odds = _pot_odds(to_call, pot)

    if community:
        score, _, hand_name = best_hand(hole, community)
        hand_rank = score[0]
    else:
        cat = _preflop_category(hole)
        hand_name = cat.replace("_", " ").title()
        hand_rank = -1

    grade = "ok"
    explanation = ""

    if action == Action.FOLD:
        if equity >= 0.65 or (community and hand_rank >= TWO_PAIR):
            grade = "mistake"
            explanation = (
                f"Folding a {hand_name} was likely wrong — your hand was strong enough to continue. "
                f"Estimated winning chance: ~{round(equity*100)}%."
            )
        elif odds is not None and equity * 100 > odds * 100 + 10:
            grade = "mistake"
            explanation = (
                f"The pot odds favored calling here (equity ~{round(equity*100)}% vs "
                f"call cost ~{round(odds*100)}%). Folding gave up value."
            )
        else:
            grade = "good"
            explanation = f"Good fold — your hand ({hand_name}) didn't justify the cost to continue."

    elif action == Action.CALL:
        if odds is not None and equity * 100 < odds * 100 - 10:
            grade = "mistake"
            explanation = (
                f"Calling here was too expensive. You're paying {round(odds*100)}% of the pot "
                f"but winning only ~{round(equity*100)}% of the time — that's a losing call over time."
            )
        elif equity >= 0.70 and to_call > 0:
            grade = "ok"
            explanation = (
                f"Calling is fine, but with {hand_name} (~{round(equity*100)}% equity), "
                f"a raise might extract more value."
            )
        else:
            grade = "good"
            explanation = f"Reasonable call — the pot odds support continuing with {hand_name}."

    elif action == Action.CHECK:
        if equity >= 0.75:
            grade = "ok"
            explanation = (
                f"Checking with {hand_name} is safe, but with ~{round(equity*100)}% equity "
                f"you're leaving value on the table. Betting builds the pot when you're ahead."
            )
        else:
            grade = "good"
            explanation = "Checking is fine here — no reason to put chips in with a marginal hand."

    elif action in (Action.RAISE, Action.ALL_IN):
        if equity <= 0.30:
            grade = "ok"
            explanation = (
                f"Raising with only ~{round(equity*100)}% equity is a bluff. Bluffs can work, "
                f"but make sure you have a good reason (position, opponent tendencies)."
            )
        elif equity >= 0.60:
            grade = "good"
            explanation = (
                f"Great raise — {hand_name} with ~{round(equity*100)}% equity means "
                f"you're ahead and building the pot correctly."
            )
        else:
            grade = "ok"
            explanation = f"Raising with {hand_name} is reasonable but slightly thin. Watch how opponents respond."

    return {"grade": grade, "explanation": explanation}


# ---------------------------------------------------------------------------
# 3. Post-hand review
# ---------------------------------------------------------------------------

def hand_review(history_decisions: list[dict], player_name: str) -> list[dict]:
    """
    Review all of the human player's decisions from a hand.
    Returns a list of coaching notes (only mistakes and notable moments).
    """
    notes = []
    player_decisions = [d for d in history_decisions if d["player"] == player_name]

    for d in player_decisions:
        action = Action[d["action"]]
        hole = d["hole_cards"]
        community = d["community"]
        amount = d["amount"]

        # Re-evaluate using a dummy pot/to_call (we stored action but not to_call in history)
        # For the review, we use equity vs action type only
        equity = _equity_estimate(hole, community)

        if community:
            score, _, hand_name = best_hand(hole, community)
            hand_rank = score[0]
        else:
            cat = _preflop_category(hole)
            hand_name = cat.replace("_", " ").title()
            hand_rank = -1

        note = None

        if action == Action.FOLD and (equity >= 0.60 or hand_rank >= TWO_PAIR):
            note = {
                "street": d["street"],
                "type": "mistake",
                "message": (
                    f"You folded {hole} ({hand_name}) on the {d['street']}. "
                    f"This hand had ~{round(equity*100)}% equity — folding here gave up value."
                ),
            }
        elif action == Action.CALL and equity <= 0.30:
            note = {
                "street": d["street"],
                "type": "mistake",
                "message": (
                    f"You called on the {d['street']} with {hole} ({hand_name}, ~{round(equity*100)}% equity). "
                    f"This was likely too loose — weak hands lose money when called into raises."
                ),
            }
        elif action in (Action.RAISE, Action.ALL_IN) and equity >= 0.65:
            note = {
                "street": d["street"],
                "type": "good",
                "message": (
                    f"Good aggression on the {d['street']}! Raising with {hand_name} "
                    f"(~{round(equity*100)}% equity) puts pressure on opponents when you're ahead."
                ),
            }

        if note:
            notes.append(note)

    if not notes:
        notes.append({
            "street": "Overall",
            "type": "info",
            "message": "No major mistakes this hand — solid play overall.",
        })

    return notes
