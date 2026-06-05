"""
Player progress persistence and adaptive learning tracker.

Saves to progress.json in the project root. Tracks 4 decision patterns
to identify the player's primary weakness and bias coaching accordingly.

Weaknesses tracked:
  preflop_loose   — called/raised with a weak pre-flop hand too often
  preflop_tight   — folded a premium/strong hand pre-flop too often
  ignored_pot_odds — called when pot odds said to fold
  too_passive     — checked or called with equity ≥ 65% (should have raised)
"""
from __future__ import annotations
import json
from pathlib import Path
from .player import Action
from .coaching import pre_action_tip, _preflop_category, _pot_odds, _equity_estimate

PROGRESS_FILE = Path(__file__).parent.parent / "progress.json"

WEAKNESS_LABELS = {
    "preflop_loose":    "Playing Too Many Hands",
    "preflop_tight":    "Folding Too Often Pre-Flop",
    "ignored_pot_odds": "Ignoring Pot Odds",
    "too_passive":      "Being Too Passive with Strong Hands",
}

WEAKNESS_TIPS = {
    "preflop_loose": (
        "You're entering too many pots with weak starting hands. "
        "Focus on the 'Your Hand' category — only play Premium, Strong, or Playable hands."
    ),
    "preflop_tight": (
        "You're folding good hands pre-flop. When you have a Premium or Strong hand, "
        "don't be afraid to raise and commit chips."
    ),
    "ignored_pot_odds": (
        "You're calling bets when the math says to fold. "
        "Check the 'Pot Odds' category before calling — if your win chance is less than the call cost, fold."
    ),
    "too_passive": (
        "You have strong hands but keep checking or just calling. "
        "When the 'Recommendation' says raise, trust it — betting builds the pot when you're ahead."
    ),
}


def _default_progress(player_name: str, chips: int) -> dict:
    return {
        "player_name": player_name,
        "chips": chips,
        "starting_chips": chips,
        "hands_played": 0,
        "hands_won": 0,
        "session_history": [],
        "decision_stats": {
            "total": 0,
            "preflop_loose": 0,
            "preflop_tight": 0,
            "ignored_pot_odds": 0,
            "too_passive": 0,
            "good_decisions": 0,
        },
        "primary_weakness": None,
        "tutorial_seen": False,
    }


def load_progress() -> dict | None:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return None
    return None


def save_progress(data: dict):
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))


def record_hand_result(data: dict, won: bool, chips_end: int):
    data["hands_played"] += 1
    if won:
        data["hands_won"] += 1
    data["chips"] = chips_end
    data["session_history"].append({
        "hand": data["hands_played"],
        "result": "won" if won else "lost",
        "chips_end": chips_end,
    })
    _update_weakness(data)
    save_progress(data)


def record_decision(
    data: dict,
    action: Action,
    hole_cards,
    community_cards,
    to_call: int,
    pot: int,
):
    """Called every time the human player makes a decision."""
    stats = data["decision_stats"]
    stats["total"] += 1

    equity = _equity_estimate(hole_cards, community_cards)
    odds = _pot_odds(to_call, pot)

    is_preflop = len(community_cards) == 0
    cat = _preflop_category(hole_cards) if is_preflop else None

    # Pattern 1: too loose pre-flop
    if is_preflop and action in (Action.CALL, Action.RAISE) and cat in ("weak", "connector"):
        stats["preflop_loose"] += 1

    # Pattern 2: too tight pre-flop
    elif is_preflop and action == Action.FOLD and cat in ("premium", "strong"):
        stats["preflop_tight"] += 1

    # Pattern 3: ignoring pot odds
    elif odds is not None and action == Action.CALL and equity * 100 < odds * 100 - 10:
        stats["ignored_pot_odds"] += 1

    # Pattern 4: too passive with strong hand
    elif action in (Action.CHECK, Action.CALL) and equity >= 0.65:
        stats["too_passive"] += 1

    # Good decision
    else:
        stats["good_decisions"] += 1

    _update_weakness(data)


def _update_weakness(data: dict):
    """Recalculate primary weakness. Requires at least 5 decisions."""
    stats = data["decision_stats"]
    total = stats["total"]
    if total < 5:
        data["primary_weakness"] = None
        return

    rates = {
        k: stats[k] / total
        for k in ("preflop_loose", "preflop_tight", "ignored_pot_odds", "too_passive")
    }
    # Only flag a weakness if it's at least 20% of decisions
    worst_key = max(rates, key=lambda k: rates[k])
    data["primary_weakness"] = worst_key if rates[worst_key] >= 0.20 else None


def get_weakness_banner(data: dict) -> dict | None:
    """Return a dict with label + tip for the player's current weakness, or None."""
    key = data.get("primary_weakness")
    if not key:
        return None
    return {
        "key": key,
        "label": WEAKNESS_LABELS[key],
        "tip": WEAKNESS_TIPS[key],
    }


def win_rate(data: dict) -> float | None:
    if data["hands_played"] == 0:
        return None
    return round(data["hands_won"] / data["hands_played"] * 100, 1)
