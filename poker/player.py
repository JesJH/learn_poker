from dataclasses import dataclass, field
from enum import Enum, auto
from .card import Card


class Action(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
    ALL_IN = auto()


class PlayerType(Enum):
    HUMAN = auto()
    AI_TIGHT = auto()       # plays few hands, bets only with strong cards
    AI_LOOSE = auto()       # plays many hands, calls often
    AI_AGGRESSIVE = auto()  # raises frequently


@dataclass
class Player:
    name: str
    chips: int
    player_type: PlayerType = PlayerType.HUMAN

    hole_cards: list[Card] = field(default_factory=list)
    bet_this_round: int = 0
    total_bet: int = 0       # total chips committed this hand
    folded: bool = False
    is_all_in: bool = False

    def reset_for_hand(self):
        self.hole_cards = []
        self.bet_this_round = 0
        self.total_bet = 0
        self.folded = False
        self.is_all_in = False

    def reset_for_round(self):
        self.bet_this_round = 0

    def place_bet(self, amount: int) -> int:
        """Deduct chips and return how much was actually bet (capped at stack)."""
        amount = min(amount, self.chips)
        self.chips -= amount
        self.bet_this_round += amount
        self.total_bet += amount
        if self.chips == 0:
            self.is_all_in = True
        return amount

    @property
    def is_active(self) -> bool:
        return not self.folded and not self.is_all_in

    @property
    def is_human(self) -> bool:
        return self.player_type == PlayerType.HUMAN

    def __repr__(self):
        return f"{self.name}(${self.chips})"
