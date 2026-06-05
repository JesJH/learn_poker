"""Smoke test: run a full hand between 3 AI players and print results."""
from poker.card import Deck
from poker.player import Player, PlayerType
from poker.game import GameState, Street
from poker.hand_evaluator import best_hand


def run_hand(players, dealer_index=0):
    game = GameState(players, small_blind=10, big_blind=20)
    game.start_hand(dealer_index)

    print(f"\n{'='*50}")
    print(f"Dealer: {players[dealer_index].name}")
    print(f"Hole cards:")
    for p in players:
        print(f"  {p.name}: {p.hole_cards}")

    # Pre-flop (BB already posted 20, UTG acts first)
    print(f"\n-- Pre-Flop --")
    current_bet = 20
    game.run_betting_round(game.first_to_act_preflop(), current_bet)
    print(f"Pot: ${game.pot}  |  Active: {[p.name for p in game.active_players()]}")

    if len(game.active_players()) > 1:
        game.deal_flop()
        print(f"\n-- Flop --  {game.community_cards}")
        game.run_betting_round(game.first_to_act_postflop(), 0)
        print(f"Pot: ${game.pot}  |  Active: {[p.name for p in game.active_players()]}")

    if len(game.active_players()) > 1:
        game.deal_turn()
        print(f"\n-- Turn --  {game.community_cards}")
        game.run_betting_round(game.first_to_act_postflop(), 0)
        print(f"Pot: ${game.pot}  |  Active: {[p.name for p in game.active_players()]}")

    if len(game.active_players()) > 1:
        game.deal_river()
        print(f"\n-- River --  {game.community_cards}")
        game.run_betting_round(game.first_to_act_postflop(), 0)
        print(f"Pot: ${game.pot}  |  Active: {[p.name for p in game.active_players()]}")

    print(f"\n-- Showdown --")
    winners = game.showdown()

    for p in game.active_players():
        score, five, name = best_hand(p.hole_cards, game.community_cards)
        print(f"  {p.name}: {p.hole_cards} -> {name} ({five})")

    game.award_pot(winners)
    print(f"\nWinner(s): {[w.name for w in winners]}")
    print(f"Chip counts: {[(p.name, p.chips) for p in players]}")
    return game


if __name__ == "__main__":
    players = [
        Player("Alice (Tight)",      chips=1000, player_type=PlayerType.AI_TIGHT),
        Player("Bob (Loose)",        chips=1000, player_type=PlayerType.AI_LOOSE),
        Player("Carol (Aggressive)", chips=1000, player_type=PlayerType.AI_AGGRESSIVE),
    ]

    print("Running 3 hands of Texas Hold'em...\n")
    for hand_num in range(3):
        print(f"\n{'#'*50}")
        print(f"  HAND {hand_num + 1}")
        run_hand(players, dealer_index=hand_num % 3)
        print(f"\nChips after hand {hand_num+1}: {[(p.name, p.chips) for p in players]}")
