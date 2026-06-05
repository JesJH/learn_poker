import random

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}  # "2"=0, "A"=12


class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.value = RANK_VALUES[rank]

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other):
        return self.value < other.value

    def __hash__(self):
        return hash((self.rank, self.suit))


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for s in SUITS for r in RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[Card]:
        if n > len(self.cards):
            raise ValueError("Not enough cards left in deck")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def __len__(self):
        return len(self.cards)
