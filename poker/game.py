from dataclasses import dataclass, field
from enum import Enum, auto
from .card import Card, Deck
from .player import Player, Action, PlayerType
from .hand_evaluator import best_hand


class Street(Enum):
    PREFLOP = "Pre-Flop"
    FLOP    = "Flop"
    TURN    = "Turn"
    RIVER   = "River"
    SHOWDOWN = "Showdown"


@dataclass
class HandHistory:
    """Records every decision made in a hand for post-hand coaching."""
    decisions: list[dict] = field(default_factory=list)

    def record(self, player_name: str, street: Street, action: Action,
               amount: int, hole_cards: list[Card], community: list[Card]):
        self.decisions.append({
            "player": player_name,
            "street": street.value,
            "action": action.name,
            "amount": amount,
            "hole_cards": list(hole_cards),
            "community": list(community),
        })


class GameState:
    """Manages one complete hand of Texas Hold'em."""

    def __init__(self, players: list[Player], small_blind: int = 10, big_blind: int = 20):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.deck = Deck()
        self.community_cards: list[Card] = []
        self.pot = 0
        self.street = Street.PREFLOP
        self.history = HandHistory()

        # Dealer button rotates each hand; caller manages this
        self.dealer_index = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def start_hand(self, dealer_index: int):
        self.dealer_index = dealer_index
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.street = Street.PREFLOP
        self.history = HandHistory()

        for p in self.players:
            p.reset_for_hand()

        self._deal_hole_cards()
        self._post_blinds()

    def _deal_hole_cards(self):
        for p in self.players:
            p.hole_cards = self.deck.deal(2)

    def _post_blinds(self):
        n = len(self.players)
        sb_index = (self.dealer_index + 1) % n
        bb_index = (self.dealer_index + 2) % n

        sb_player = self.players[sb_index]
        bb_player = self.players[bb_index]

        self.pot += sb_player.place_bet(self.small_blind)
        self.pot += bb_player.place_bet(self.big_blind)

    # ------------------------------------------------------------------
    # Betting round
    # ------------------------------------------------------------------

    def run_betting_round(self, first_to_act: int, current_bet: int = 0) -> int:
        """Run one full betting round. Returns the final bet level."""
        n = len(self.players)
        players_acted = set()
        i = first_to_act

        while True:
            player = self.players[i % n]

            if not player.is_active:
                i += 1
                # Everyone folded or all-in — nothing left to do
                active = [p for p in self.players if not p.folded]
                if len(active) <= 1:
                    break
                all_matched = all(
                    p.bet_this_round == current_bet or not p.is_active
                    for p in self.players if not p.folded
                )
                if all_matched and len(players_acted) >= len([p for p in self.players if not p.folded]):
                    break
                continue

            to_call = current_bet - player.bet_this_round

            if player.is_human:
                action, amount = self._get_human_action(player, to_call, current_bet)
            else:
                action, amount = self._get_ai_action(player, to_call, current_bet)

            self.history.record(player.name, self.street, action, amount,
                                player.hole_cards, self.community_cards)

            if action == Action.FOLD:
                player.folded = True
            elif action in (Action.CHECK, Action.CALL):
                if to_call > 0:
                    self.pot += player.place_bet(to_call)
            elif action in (Action.RAISE, Action.ALL_IN):
                self.pot += player.place_bet(amount)
                current_bet = player.bet_this_round
                players_acted = {player.name}  # others must respond to raise

            players_acted.add(player.name)

            # Check if only one player remains
            active = [p for p in self.players if not p.folded]
            if len(active) <= 1:
                break

            # Check if all active players have matched the current bet
            all_matched = all(
                p.bet_this_round == current_bet or p.is_all_in
                for p in self.players if not p.folded
            )
            all_have_acted = all(
                p.name in players_acted or p.folded or p.is_all_in
                for p in self.players
            )
            if all_matched and all_have_acted:
                break

            i += 1

        return current_bet

    def _get_human_action(self, player: Player, to_call: int, current_bet: int) -> tuple[Action, int]:
        """Placeholder — Streamlit UI will override this."""
        raise NotImplementedError("Human action must be provided by the UI layer")

    def _get_ai_action(self, player: Player, to_call: int, current_bet: int) -> tuple[Action, int]:
        """Delegate to the AI module."""
        from .ai import decide_action
        return decide_action(player, to_call, current_bet, self.community_cards, self.pot)

    # ------------------------------------------------------------------
    # Street progression
    # ------------------------------------------------------------------

    def deal_flop(self):
        self.deck.deal(1)  # burn card
        self.community_cards += self.deck.deal(3)
        self.street = Street.FLOP
        for p in self.players:
            p.reset_for_round()

    def deal_turn(self):
        self.deck.deal(1)
        self.community_cards += self.deck.deal(1)
        self.street = Street.TURN
        for p in self.players:
            p.reset_for_round()

    def deal_river(self):
        self.deck.deal(1)
        self.community_cards += self.deck.deal(1)
        self.street = Street.RIVER
        for p in self.players:
            p.reset_for_round()

    # ------------------------------------------------------------------
    # Showdown
    # ------------------------------------------------------------------

    def showdown(self) -> list[Player]:
        """Determine winner(s). Returns list of winners (tie possible)."""
        self.street = Street.SHOWDOWN
        active = [p for p in self.players if not p.folded]

        if len(active) == 1:
            return active  # everyone else folded

        scored = []
        for p in active:
            score, five, name = best_hand(p.hole_cards, self.community_cards)
            scored.append((score, p, name))

        best_score = max(s[0] for s in scored)
        winners = [(p, name) for score, p, name in scored if score == best_score]
        return [w[0] for w in winners]

    def award_pot(self, winners: list[Player]):
        share = self.pot // len(winners)
        remainder = self.pot % len(winners)
        for i, w in enumerate(winners):
            w.chips += share + (remainder if i == 0 else 0)
        self.pot = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def active_players(self) -> list[Player]:
        return [p for p in self.players if not p.folded]

    def first_to_act_preflop(self) -> int:
        """UTG (under the gun) acts first pre-flop: 3 seats left of dealer."""
        return (self.dealer_index + 3) % len(self.players)

    def first_to_act_postflop(self) -> int:
        """Small blind (or next active player) acts first post-flop."""
        n = len(self.players)
        start = (self.dealer_index + 1) % n
        for offset in range(n):
            p = self.players[(start + offset) % n]
            if not p.folded:
                return (start + offset) % n
        return start
