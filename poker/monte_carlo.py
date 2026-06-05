"""
Monte Carlo equity calculator (Option A: random opponent hands).

For each simulation:
  1. Build remaining deck (exclude known cards)
  2. Deal random hands to each opponent
  3. Run out remaining community cards randomly
  4. Evaluate all hands — count wins / ties / losses

This is the same structure used in options pricing Monte Carlo simulations:
sample many possible future states, compute the outcome in each, take the
expectation. Here "future states" = possible opponent hands + board runouts.
"""
from __future__ import annotations
import random
from .card import Card, SUITS, RANKS
from .hand_evaluator import best_hand


def run(
    hole_cards: list[Card],
    community_cards: list[Card],
    num_opponents: int,
    num_simulations: int = 1000,
) -> dict:
    """
    Returns:
        win_pct   : float  (0-1)
        tie_pct   : float  (0-1)
        lose_pct  : float  (0-1)
        n         : int    (simulations actually completed)
        std_error : float  (95% CI half-width on win_pct)
    """
    known = {(c.rank, c.suit) for c in hole_cards + community_cards}
    remaining = [Card(r, s) for s in SUITS for r in RANKS if (r, s) not in known]

    cards_needed_on_board = 5 - len(community_cards)
    cards_per_sim = num_opponents * 2 + cards_needed_on_board

    # If not enough cards in deck, reduce simulations to available combos
    if len(remaining) < cards_per_sim:
        return {"win_pct": 0, "tie_pct": 0, "lose_pct": 1, "n": 0, "std_error": 0}

    wins = ties = losses = 0

    for _ in range(num_simulations):
        sample = random.sample(remaining, cards_per_sim)

        # Split sample into opponent hands and board completion
        opp_hands = [sample[i * 2: i * 2 + 2] for i in range(num_opponents)]
        board = community_cards + sample[num_opponents * 2:]

        my_score = best_hand(hole_cards, board)[0]
        opp_scores = [best_hand(hand, board)[0] for hand in opp_hands]
        best_opp = max(opp_scores)

        if my_score > best_opp:
            wins += 1
        elif my_score == best_opp:
            ties += 1
        else:
            losses += 1

    n = num_simulations
    win_pct = wins / n
    # Standard error of the proportion — used to show confidence interval
    import math
    std_error = math.sqrt(win_pct * (1 - win_pct) / n) * 1.96  # 95% CI

    return {
        "win_pct": round(win_pct, 4),
        "tie_pct": round(ties / n, 4),
        "lose_pct": round(losses / n, 4),
        "n": n,
        "std_error": round(std_error, 4),
    }
