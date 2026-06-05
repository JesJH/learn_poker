"""
Kuhn Poker — a 3-card toy game used to teach Game Theory Optimal (GTO) play.

Rules:
  - Deck: J, Q, K (Jack=0, Queen=1, King=2)
  - 2 players, each antes 1 chip
  - Player 1 acts first: Check or Bet (1 chip)
  - If P1 checked: P2 can Check (showdown) or Bet
    - If P2 bet: P1 can Call or Fold
  - If P1 bet: P2 can Call or Fold
  - Higher card wins at showdown

Nash Equilibrium (analytically derived — no algorithm needed):
  Player 1:
    King  → always Bet
    Queen → always Check
    Jack  → Bet with probability 1/3 (bluff), Check with prob 2/3

  Player 2 (facing a bet):
    King  → always Call
    Queen → Call with probability 1/3, Fold with prob 2/3
    Jack  → always Fold

  Game value: -1/18 chips per hand for Player 1 (P2 has slight structural edge)

Quant tie-in:
  Nash Equilibrium = unexploitable strategy. In markets, this is the strategy
  a perfectly rational market maker would use. Deviating from it creates
  exploitable edges — the same as "alpha" in statistical arbitrage.

References:
  Kuhn, H.W. (1950). "A Simplified Two-Person Poker." Annals of Mathematics Studies.
  Chen & Ankenman, The Mathematics of Poker (2006), Ch. 6.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

CARDS = ["J", "Q", "K"]
CARD_VALUE = {"J": 0, "Q": 1, "K": 2}

# GTO strategy: (card, situation) -> action probabilities
# situation: "first_action" | "facing_bet"
GTO_P1 = {
    "K": {"check": 0.0,  "bet": 1.0},   # always bet
    "Q": {"check": 1.0,  "bet": 0.0},   # always check
    "J": {"check": 2/3,  "bet": 1/3},   # bluff 1/3
}
GTO_P2_FACING_BET = {
    "K": {"call": 1.0,  "fold": 0.0},   # always call
    "Q": {"call": 1/3,  "fold": 2/3},   # defend 1/3
    "J": {"call": 0.0,  "fold": 1.0},   # always fold
}
GTO_P2_FACING_CHECK = {
    "K": {"check": 0.0,  "bet": 1.0},   # always bet
    "Q": {"check": 1.0,  "bet": 0.0},   # always check
    "J": {"check": 2/3,  "bet": 1/3},   # bluff 1/3
}

# Exploitative (loose) bot — calls too much, bluffs too much
LOOSE_P1 = {
    "K": {"check": 0.0,  "bet": 1.0},
    "Q": {"check": 0.3,  "bet": 0.7},   # over-bets Q
    "J": {"check": 0.3,  "bet": 0.7},   # over-bluffs J
}
LOOSE_P2_FACING_BET = {
    "K": {"call": 1.0,  "fold": 0.0},
    "Q": {"call": 0.8,  "fold": 0.2},   # calls too wide
    "J": {"call": 0.4,  "fold": 0.6},   # calls too wide
}


@dataclass
class KuhnHand:
    p1_card: str
    p2_card: str
    pot: int = 2          # both antes
    p1_action: str = ""   # "check" or "bet"
    p2_action: str = ""   # "check"/"call"/"fold"/"bet"
    p1_response: str = "" # "call" or "fold" (if P2 bet after P1 checked)
    history: list[str] = field(default_factory=list)
    result: dict | None = None

    def winner(self) -> str:
        if CARD_VALUE[self.p1_card] > CARD_VALUE[self.p2_card]:
            return "P1"
        return "P2"


def deal() -> KuhnHand:
    cards = random.sample(CARDS, 2)
    return KuhnHand(p1_card=cards[0], p2_card=cards[1])


def gto_action(strategy: dict, card: str) -> str:
    probs = strategy[card]
    actions = list(probs.keys())
    weights = list(probs.values())
    return random.choices(actions, weights=weights)[0]


def resolve(hand: KuhnHand, p1_action: str, p2_action: str, p1_response: str = "") -> dict:
    """
    Resolve a Kuhn Poker hand given all actions.
    Returns: { winner, p1_profit, p2_profit, outcome_description }
    """
    pot = 2  # antes

    if p1_action == "bet":
        pot += 1  # P1 bets
        if p2_action == "fold":
            return {"winner": "P1", "p1_profit": 1, "p2_profit": -1,
                    "desc": "P2 folded to P1's bet. P1 wins the antes."}
        else:  # call
            pot += 1  # P2 calls
            winner = hand.winner()
            profit = pot // 2  # winner gets both antes + both bets = 2 chips net
            return {
                "winner": winner,
                "p1_profit": profit if winner == "P1" else -profit,
                "p2_profit": profit if winner == "P2" else -profit,
                "desc": f"P2 called. {winner} wins at showdown ({hand.p1_card} vs {hand.p2_card}).",
            }
    else:  # P1 checked
        if p2_action == "check":
            winner = hand.winner()
            return {
                "winner": winner,
                "p1_profit": 1 if winner == "P1" else -1,
                "p2_profit": 1 if winner == "P2" else -1,
                "desc": f"Both checked. {winner} wins at showdown ({hand.p1_card} vs {hand.p2_card}).",
            }
        else:  # P2 bet
            pot += 1
            if p1_response == "fold":
                return {"winner": "P2", "p1_profit": -1, "p2_profit": 1,
                        "desc": "P1 folded to P2's bet. P2 wins."}
            else:  # P1 calls
                pot += 1
                winner = hand.winner()
                profit = 2
                return {
                    "winner": winner,
                    "p1_profit": profit if winner == "P1" else -profit,
                    "p2_profit": profit if winner == "P2" else -profit,
                    "desc": f"P1 called P2's bet. {winner} wins at showdown ({hand.p1_card} vs {hand.p2_card}).",
                }


def compute_strategy_ev(p1_strategy: dict, p2_bet_strategy: dict, num_samples: int = 5000) -> float:
    """
    Monte Carlo EV for P1 using given strategies.
    Returns average chips won per hand by P1.
    """
    total = 0
    for _ in range(num_samples):
        hand = deal()
        a1 = gto_action(p1_strategy, hand.p1_card)
        if a1 == "bet":
            a2 = gto_action(p2_bet_strategy, hand.p2_card)
            r = resolve(hand, "bet", a2)
        else:
            # P2 checks or bets — use same strategy for P2's bet decision
            a2 = gto_action(GTO_P2_FACING_CHECK, hand.p2_card)
            if a2 == "bet":
                # P1 response — use GTO: call with K/Q, fold with J
                if hand.p1_card == "K":
                    resp = "call"
                elif hand.p1_card == "J":
                    resp = "fold"
                else:
                    resp = random.choice(["call", "fold"])
                r = resolve(hand, "check", "bet", resp)
            else:
                r = resolve(hand, "check", "check")
        total += r["p1_profit"]
    return round(total / num_samples, 4)


def gto_vs_exploitative_summary() -> dict:
    """Run both strategies and compare EVs."""
    ev_gto = compute_strategy_ev(GTO_P1, GTO_P2_FACING_BET)
    ev_loose = compute_strategy_ev(LOOSE_P1, LOOSE_P2_FACING_BET)
    return {
        "ev_gto": ev_gto,
        "ev_loose": ev_loose,
        "ev_diff": round(ev_loose - ev_gto, 4),
        "note": (
            "GTO is the Nash Equilibrium — unexploitable but not maximally exploitative. "
            "The loose strategy deviates from equilibrium. Against a GTO opponent it loses more; "
            "against another loose opponent it may win more. This is the core tension: "
            "GTO = market efficiency. Exploitative = finding alpha against a mispricing opponent."
        ),
    }
