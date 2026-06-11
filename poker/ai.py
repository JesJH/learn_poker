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


def _min_raise(current_bet: int, big_blind: int) -> int:
    """Texas Hold'em minimum raise: size of the previous bet/raise."""
    return max(current_bet * 2, big_blind)


def decide_action(player: Player, to_call: int, current_bet: int,
                  community: list[Card], pot: int,
                  big_blind: int = 20) -> tuple[Action, int]:
    """Choose an action for an AI player."""
    strength = _strength(player, community)

    if player.player_type == PlayerType.AI_TIGHT:
        return _tight(player, strength, to_call, current_bet, pot, big_blind)
    elif player.player_type == PlayerType.AI_LOOSE:
        return _loose(player, strength, to_call, current_bet, pot, big_blind)
    elif player.player_type == PlayerType.AI_AGGRESSIVE:
        return _aggressive(player, strength, to_call, current_bet, pot, big_blind)
    else:
        return _tight(player, strength, to_call, current_bet, pot, big_blind)


def _tight(player: Player, strength: float, to_call: int, current_bet: int,
           pot: int, big_blind: int = 20):
    """Plays top ~30% of hands; folds otherwise. Occasionally slow-plays strong hands."""
    min_r = _min_raise(current_bet, big_blind)

    if strength >= 0.80:
        # Occasionally slow-play a monster to trap opponents
        if random.random() < 0.20:
            if to_call == 0:
                return (Action.CHECK, 0)
            if to_call <= player.chips // 3:
                return (Action.CALL, to_call)
        raise_amount = int(current_bet * random.uniform(2.0, 2.8)) + random.randint(0, big_blind)
        return (Action.RAISE, min(max(raise_amount, min_r), player.chips))

    if strength >= 0.60:
        # Mix of raise and call — don't always telegraph strength
        if random.random() < 0.50:
            raise_amount = int(current_bet * random.uniform(1.5, 2.2)) + random.randint(0, big_blind)
            return (Action.RAISE, min(max(raise_amount, min_r), player.chips))
        if to_call == 0:
            return (Action.CHECK, 0)
        if to_call <= player.chips // 4:
            return (Action.CALL, to_call)

    if strength >= 0.45 and to_call <= player.chips // 5:
        return (Action.CALL, to_call)
    if to_call == 0:
        return (Action.CHECK, 0)
    return (Action.FOLD, 0)


def _loose(player: Player, strength: float, to_call: int, current_bet: int,
           pot: int, big_blind: int = 20):
    """Calls with most hands; raises occasionally on strong hands; folds big bets with air."""
    min_r = _min_raise(current_bet, big_blind)

    if strength >= 0.80:
        if random.random() < 0.55:
            raise_amount = int(current_bet + pot // random.randint(2, 4)) + random.randint(0, big_blind)
            return (Action.RAISE, min(max(raise_amount, min_r), player.chips))

    # Plays ~70% of hands; folds the worst to large bets
    if strength >= 0.35 or random.random() < 0.25:
        if to_call == 0:
            return (Action.CHECK, 0)
        if to_call <= player.chips // 2:
            return (Action.CALL, to_call)

    if to_call == 0:
        return (Action.CHECK, 0)
    if strength < 0.25 and to_call > player.chips // 4:
        return (Action.FOLD, 0)
    if to_call <= player.chips // 3:
        return (Action.CALL, to_call)
    return (Action.FOLD, 0)


def _aggressive(player: Player, strength: float, to_call: int, current_bet: int,
                pot: int, big_blind: int = 20):
    """Raises frequently with varied sizing; bluffs occasionally; avoids constant all-ins."""
    bluff = random.random() < 0.12
    min_r = _min_raise(current_bet, big_blind)

    if strength >= 0.80:
        # Strong hand — raise meaningfully but not always all-in
        raise_amount = int(current_bet * random.uniform(2.2, 3.0)) + random.randint(0, pot // 4)
        return (Action.RAISE, min(max(raise_amount, min_r), player.chips))

    if strength >= 0.55 or bluff:
        if random.random() < 0.70:  # raise 70% of the time, call 30%
            raise_amount = int(current_bet * random.uniform(1.5, 2.2)) + random.randint(0, big_blind * 2)
            # Cap at 55% of stack so the aggressive player survives more hands
            raise_amount = min(raise_amount, player.chips * 55 // 100)
            raise_amount = max(raise_amount, min_r)
            if raise_amount >= player.chips or raise_amount <= to_call:
                return (Action.CALL, to_call) if to_call > 0 else (Action.CHECK, 0)
            return (Action.RAISE, raise_amount)
        else:
            if to_call == 0:
                return (Action.CHECK, 0)
            if to_call <= player.chips // 3:
                return (Action.CALL, to_call)

    if to_call == 0:
        return (Action.CHECK, 0)
    if to_call <= player.chips // 4:
        return (Action.CALL, to_call)
    return (Action.FOLD, 0)
