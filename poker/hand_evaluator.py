from __future__ import annotations
from itertools import combinations
from collections import Counter
from .card import Card

# Hand rank constants (higher = better)
HIGH_CARD      = 0
ONE_PAIR       = 1
TWO_PAIR       = 2
THREE_OF_A_KIND = 3
STRAIGHT       = 4
FLUSH          = 5
FULL_HOUSE     = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
ROYAL_FLUSH    = 9

HAND_NAMES = {
    HIGH_CARD: "High Card",
    ONE_PAIR: "One Pair",
    TWO_PAIR: "Two Pair",
    THREE_OF_A_KIND: "Three of a Kind",
    STRAIGHT: "Straight",
    FLUSH: "Flush",
    FULL_HOUSE: "Full House",
    FOUR_OF_A_KIND: "Four of a Kind",
    STRAIGHT_FLUSH: "Straight Flush",
    ROYAL_FLUSH: "Royal Flush",
}


def _values(cards: list[Card]) -> list[int]:
    return sorted([c.value for c in cards], reverse=True)


def _is_flush(cards: list[Card]) -> bool:
    return len({c.suit for c in cards}) == 1


def _straight_high(values: list[int]) -> int | None:
    """Return the high card value if cards form a straight, else None.
    Handles the wheel (A-2-3-4-5) by treating A as value -1 in that case."""
    vals = sorted(set(values), reverse=True)
    if len(vals) < 5:
        return None
    if vals[0] - vals[4] == 4:
        return vals[0]
    # Wheel: A-2-3-4-5 (A treated as low)
    if vals == [12, 3, 2, 1, 0]:
        return 3  # 5-high straight
    return None


def _score_five(cards: list[Card]) -> tuple:
    """Return a comparable tuple that ranks a 5-card hand."""
    vals = _values(cards)
    counts = Counter(vals)
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    group_counts = [g[1] for g in groups]
    group_vals  = [g[0] for g in groups]

    flush = _is_flush(cards)
    straight_high = _straight_high(vals)

    if flush and straight_high is not None:
        rank = ROYAL_FLUSH if straight_high == 12 else STRAIGHT_FLUSH
        return (rank, straight_high)

    if group_counts[0] == 4:
        return (FOUR_OF_A_KIND, group_vals[0], group_vals[1])

    if group_counts[:2] == [3, 2]:
        return (FULL_HOUSE, group_vals[0], group_vals[1])

    if flush:
        return (FLUSH, *vals)

    if straight_high is not None:
        return (STRAIGHT, straight_high)

    if group_counts[0] == 3:
        return (THREE_OF_A_KIND, group_vals[0], *group_vals[1:])

    if group_counts[:2] == [2, 2]:
        pairs = sorted(group_vals[:2], reverse=True)
        return (TWO_PAIR, pairs[0], pairs[1], group_vals[2])

    if group_counts[0] == 2:
        return (ONE_PAIR, group_vals[0], *group_vals[1:])

    return (HIGH_CARD, *vals)


def best_hand(hole_cards: list[Card], community_cards: list[Card]) -> tuple[tuple, list[Card], str]:
    """Find the best 5-card hand from up to 7 cards.

    Returns:
        score: comparable tuple (higher = better)
        best_five: the 5 cards making the best hand
        name: human-readable hand name
    """
    all_cards = hole_cards + community_cards
    best_score = None
    best_five = None

    for five in combinations(all_cards, 5):
        score = _score_five(list(five))
        if best_score is None or score > best_score:
            best_score = score
            best_five = list(five)

    name = HAND_NAMES[best_score[0]]
    return best_score, best_five, name


def hand_strength(hole_cards: list[Card], community_cards: list[Card]) -> float:
    """Quick 0-1 strength estimate based solely on hand rank (not equity).
    Used for coaching hints, not game logic."""
    score, _, _ = best_hand(hole_cards, community_cards)
    rank = score[0]
    # Map hand rank to rough percentile: HIGH_CARD=0, ROYAL_FLUSH=1
    return round(rank / ROYAL_FLUSH, 2)
