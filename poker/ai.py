"""
Rule-based AI opponents with three distinct personalities.

Tight:      only plays strong hands, rarely bluffs
Loose:      plays many hands, calls frequently
Aggressive: raises often, puts pressure on opponents
"""
import random
from .card import Card
from .player import Player, Action, PlayerType
from .hand_evaluator import hand_strength, best_hand


def _strength(player: Player, community: list[Card]) -> float:
    """0-1 hand strength from current cards."""
    if not community:
        return _preflop_strength(player.hole_cards)
    return hand_strength(player.hole_cards, community)


def _preflop_strength(hole: list[Card]) -> float:
    """Simple pre-flop strength estimate without community cards."""
    if len(hole) < 2:
        return 0.0
    a, b = hole[0], hole[1]
    high = max(a.value, b.value)
    low = min(a.value, b.value)
    paired = a.rank == b.rank
    suited = a.suit == b.suit
    connected = abs(a.value - b.value) == 1

    score = high / 12.0  # normalize 0-1

    if paired:
        score += 0.3
    if suited:
        score += 0.1
    if connected:
        score += 0.05

    return min(score, 1.0)


def decide_action(player: Player, to_call: int, current_bet: int,
                  community: list[Card], pot: int) -> tuple[Action, int]:
    """Choose an action for an AI player."""
    strength = _strength(player, community)

    if player.player_type == PlayerType.AI_TIGHT:
        return _tight(player, strength, to_call, current_bet, pot)
    elif player.player_type == PlayerType.AI_LOOSE:
        return _loose(player, strength, to_call, current_bet, pot)
    elif player.player_type == PlayerType.AI_AGGRESSIVE:
        return _aggressive(player, strength, to_call, current_bet, pot)
    else:
        return _tight(player, strength, to_call, current_bet, pot)


def _tight(player: Player, strength: float, to_call: int, current_bet: int, pot: int):
    """Only plays top ~30% of hands, folds otherwise."""
    if strength >= 0.7:
        raise_amount = current_bet * 2 + random.randint(0, pot // 4)
        return (Action.RAISE, min(raise_amount, player.chips))
    if strength >= 0.45 and to_call <= player.chips // 5:
        return (Action.CALL, to_call)
    if to_call == 0:
        return (Action.CHECK, 0)
    return (Action.FOLD, 0)


def _loose(player: Player, strength: float, to_call: int, current_bet: int, pot: int):
    """Calls with most hands, rarely raises or folds."""
    if strength >= 0.8:
        raise_amount = current_bet + pot // 3
        return (Action.RAISE, min(raise_amount, player.chips))
    if to_call == 0:
        return (Action.CHECK, 0)
    if to_call <= player.chips // 2:
        return (Action.CALL, to_call)
    if strength >= 0.5:
        return (Action.CALL, to_call)
    return (Action.FOLD, 0)


def _aggressive(player: Player, strength: float, to_call: int, current_bet: int, pot: int):
    """Raises frequently, uses position/pot pressure, bluffs occasionally."""
    bluff = random.random() < 0.15  # 15% bluff frequency

    if strength >= 0.55 or bluff:
        raise_amount = max(current_bet * 2, pot // 2) + random.randint(0, 20)
        raise_amount = min(raise_amount, player.chips)
        return (Action.RAISE, raise_amount)
    if to_call == 0:
        return (Action.CHECK, 0)
    if to_call <= player.chips // 3:
        return (Action.CALL, to_call)
    return (Action.FOLD, 0)
