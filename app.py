"""
Learn to Play Poker — Streamlit app.

State machine phases (stored in st.session_state):
  setup          → name entry, load/new progress, chip config
  tutorial       → skippable hand-rankings + terms reference
  hand_start     → deal cards, post blinds
  betting        → human waits for input; AI actions run automatically
  showdown       → display results, hand review coaching
  hand_over_no_showdown → everyone folded, no showdown needed
  game_over      → player is out of chips
"""
import streamlit as st
from poker.player import Player, PlayerType, Action
from poker.game import GameState, Street
from poker.ai import decide_action as ai_decide
from poker.hand_evaluator import best_hand
from poker.coaching import pre_action_tip, evaluate_action, hand_review
from poker.progress import (
    load_progress, save_progress, record_hand_result, record_decision,
    get_weakness_banner, win_rate, _default_progress,
)
from poker import monte_carlo, kelly, ev_calculator
from poker.quant_concepts import concept_for_hand, get_concept
from poker import llm_coach
from poker import kuhn_poker

st.set_page_config(page_title="Learn to Play Poker", page_icon="🃏", layout="wide")

# ---------------------------------------------------------------------------
# Card display helpers
# ---------------------------------------------------------------------------

SUIT_COLOR = {"♠": "#1a1a1a", "♥": "#cc0000", "♦": "#cc0000", "♣": "#1a1a1a"}


def card_html(card, hidden=False) -> str:
    if hidden:
        return (
            '<span style="display:inline-block;background:#1a5276;color:white;'
            'border:2px solid #ccc;border-radius:6px;padding:6px 10px;'
            'font-size:1.4rem;margin:3px;min-width:42px;text-align:center;">🂠</span>'
        )
    color = SUIT_COLOR.get(card.suit, "#000")
    return (
        f'<span style="display:inline-block;background:white;color:{color};'
        f'border:2px solid #ccc;border-radius:6px;padding:6px 10px;'
        f'font-size:1.4rem;margin:3px;min-width:42px;text-align:center;'
        f'box-shadow:1px 1px 3px rgba(0,0,0,0.2);">'
        f'{card.rank}{card.suit}</span>'
    )


def render_cards(cards, hidden=False):
    html = "".join(card_html(c, hidden) for c in cards)
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: reference card + progress stats
# ---------------------------------------------------------------------------

HAND_RANKINGS = [
    ("Royal Flush",     "A♠ K♠ Q♠ J♠ 10♠", "Best possible hand — Ace-high straight flush"),
    ("Straight Flush",  "9♥ 8♥ 7♥ 6♥ 5♥",  "Five consecutive cards of the same suit"),
    ("Four of a Kind",  "K♠ K♥ K♦ K♣ 3♠",  "Four cards of the same rank"),
    ("Full House",      "J♠ J♥ J♦ 8♣ 8♥",  "Three of a kind + a pair"),
    ("Flush",           "A♦ J♦ 8♦ 5♦ 2♦",  "Any five cards of the same suit"),
    ("Straight",        "10♠ 9♥ 8♦ 7♣ 6♠", "Five consecutive cards, mixed suits"),
    ("Three of a Kind", "7♠ 7♥ 7♦ K♣ 2♠",  "Three cards of the same rank"),
    ("Two Pair",        "Q♠ Q♦ 4♥ 4♣ A♠",  "Two different pairs"),
    ("One Pair",        "10♠ 10♥ A♦ 7♣ 3♠","Two cards of the same rank"),
    ("High Card",       "A♠ J♦ 8♥ 5♣ 2♠",  "No combination — highest card wins"),
]

POKER_TERMS = {
    "Blinds":      "Forced bets posted before cards are dealt. Small blind = half of big blind.",
    "Pre-Flop":    "The first betting round, after each player receives 2 hole cards.",
    "Flop":        "The first 3 community cards dealt face-up. Triggers a betting round.",
    "Turn":        "The 4th community card. Another betting round follows.",
    "River":       "The 5th and final community card. Last betting round before showdown.",
    "Check":       "Pass the action without betting (only when no one has bet yet).",
    "Call":        "Match the current bet to stay in the hand.",
    "Raise":       "Increase the current bet, forcing others to call or fold.",
    "Fold":        "Discard your hand and forfeit the pot.",
    "Pot Odds":    "The ratio of the call amount to the total pot. Guides whether calling is profitable.",
    "Position":    "Where you sit relative to the dealer. Acting last is an advantage.",
    "Equity":      "Your estimated probability of winning the hand.",
    "All-In":      "Bet all remaining chips.",
    "Showdown":    "When remaining players reveal cards to determine the winner.",
}


def render_sidebar(progress: dict | None):
    with st.sidebar:
        st.title("📖 Reference Card")

        if progress:
            st.divider()
            st.subheader("Your Progress")
            col1, col2 = st.columns(2)
            col1.metric("Chips", f"${progress['chips']}")
            col2.metric("Hands", progress["hands_played"])
            wr = win_rate(progress)
            if wr is not None:
                st.metric("Win Rate", f"{wr}%")
            weakness = get_weakness_banner(progress)
            if weakness:
                st.warning(f"**Focus area:** {weakness['label']}\n\n{weakness['tip']}")

        st.divider()
        with st.expander("🏆 Hand Rankings (best → worst)", expanded=False):
            for rank, (name, example, desc) in enumerate(HAND_RANKINGS, 1):
                st.markdown(f"**{rank}. {name}**")
                st.caption(f"`{example}` — {desc}")

        with st.expander("📚 Poker Terms", expanded=False):
            for term, definition in POKER_TERMS.items():
                st.markdown(f"**{term}:** {definition}")

        with st.expander("🃏 How a Hand Works", expanded=False):
            st.markdown("""
1. **Blinds posted** — small blind & big blind put chips in before cards
2. **Hole cards dealt** — each player gets 2 private cards
3. **Pre-flop betting** — act in order around the table
4. **Flop** — 3 community cards revealed; another betting round
5. **Turn** — 4th community card; another betting round
6. **River** — 5th community card; final betting round
7. **Showdown** — remaining players reveal hands; best hand wins the pot

Your 2 hole cards + the 5 community cards = 7 cards total.
You use the **best 5 of those 7** to make your hand.
""")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "phase": "setup",
        "game": None,
        "dealer_index": 0,
        "current_bet": 0,
        "action_queue": [],
        "human_index": 0,
        "last_tip": None,
        "show_tip": False,
        "last_feedback": None,
        "hand_review_notes": [],
        "street_log": [],
        "hand_number": 0,
        "to_call": 0,
        "progress": None,
        # Quant mode
        "mode": "standard",                 # "standard" or "quant"
        "quant_equity": None,               # latest Monte Carlo result dict
        "quant_ev": None,                   # latest EV calculation dict
        "quant_kelly": None,                # latest Kelly result dict
        "current_concept": None,            # concept introduced this hand
        "pending_challenge": None,          # challenge question awaiting answer
        "challenge_answer": "",             # player's typed answer
        "challenge_feedback": None,         # LLM feedback on their answer
        "show_quant": False,
        "kuhn_hand": None,
        "kuhn_step": "p1_act",
        "kuhn_p2_bet": False,
        "kuhn_log": [],
        "kuhn_ev_summary": None,
        "ask_coach_answer": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
ss = st.session_state

# ---------------------------------------------------------------------------
# Phase: setup
# ---------------------------------------------------------------------------

def show_setup():
    st.title("🃏 Learn to Play Poker")
    st.markdown("### Texas Hold'em — play hands, get coaching, improve over time")
    st.divider()

    saved = load_progress()

    if saved:
        st.success(f"Welcome back, **{saved['player_name']}**! "
                   f"You have **${saved['chips']}** chips and have played "
                   f"**{saved['hands_played']}** hands.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Continue where I left off", type="primary", use_container_width=True):
                _start_game(saved["player_name"], saved["chips"],
                            small_blind=10, progress=saved)
        with col2:
            if st.button("🔄 Start fresh", use_container_width=True):
                PROGRESS_FILE_PATH = __import__("pathlib").Path("progress.json")
                if PROGRESS_FILE_PATH.exists():
                    PROGRESS_FILE_PATH.unlink()
                st.rerun()
        return

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Your name", value="Player", max_chars=20)
        chips = st.number_input("Starting chips", min_value=200, max_value=10000,
                                value=1000, step=100)
        small_blind = st.number_input("Small blind", min_value=5, value=10, step=5)
        st.caption(f"Big blind: ${small_blind * 2}")

        st.subheader("Game Mode")
        mode = st.radio(
            "Choose your mode",
            options=["standard", "quant"],
            format_func=lambda m: {
                "standard": "🃏 Standard — coaching focused on poker strategy",
                "quant": "🔬 Quant Trading — EV, Kelly Criterion, and trading theory",
            }[m],
            label_visibility="collapsed",
        )

    with col2:
        st.subheader("Your Opponents")
        st.markdown("""
- **Alex (Tight)** — only plays strong hands, won't bluff often
- **Blake (Loose)** — calls almost everything, hard to push off hands
- **Casey (Aggressive)** — raises frequently, puts pressure on you
        """)
        show_tut = st.checkbox("Show tutorial before first hand", value=True)
        if mode == "quant":
            st.info(
                "**Quant mode** teaches one concept per hand — Expected Value, "
                "Kelly Criterion, Bayesian Updating, and more. After learning each concept, "
                "you'll be asked to apply it on the next hand and explain your reasoning."
            )

    st.divider()
    col_start, col_kuhn = st.columns(2)
    with col_start:
        if st.button("▶ Start Game", type="primary", use_container_width=True):
            progress = _default_progress(player_name, chips)
            progress["tutorial_seen"] = not show_tut
            save_progress(progress)
            ss.mode = mode
            _start_game(player_name, chips, small_blind, progress)
    with col_kuhn:
        if st.button("🔬 Kuhn Poker Lab (GTO Theory)", use_container_width=True):
            ss.mode = "quant"
            ss.phase = "kuhn_lab"
            ss.kuhn_hand = None
            ss.kuhn_log = []
            st.rerun()


def _start_game(player_name: str, chips: int, small_blind: int, progress: dict):
    human = Player(player_name, chips=chips, player_type=PlayerType.HUMAN)
    opponents = [
        Player("Alex (Tight)",       chips=chips, player_type=PlayerType.AI_TIGHT),
        Player("Blake (Loose)",      chips=chips, player_type=PlayerType.AI_LOOSE),
        Player("Casey (Aggressive)", chips=chips, player_type=PlayerType.AI_AGGRESSIVE),
    ]
    players = [human] + opponents
    ss.human_index = 0

    game = GameState(players, small_blind=small_blind, big_blind=small_blind * 2)
    ss.game = game
    ss.dealer_index = 0
    ss.hand_number = 0
    ss.progress = progress

    if not progress.get("tutorial_seen"):
        ss.phase = "tutorial"
    else:
        ss.phase = "hand_start"
    st.rerun()


# ---------------------------------------------------------------------------
# Phase: tutorial
# ---------------------------------------------------------------------------

def show_tutorial():
    st.title("🃏 Quick Start Guide")
    st.markdown("Learn the basics before your first hand. You can always reopen this from the sidebar.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Hand Rankings")
        st.caption("Best hand wins at showdown. Higher = better.")
        for rank, (name, example, desc) in enumerate(HAND_RANKINGS, 1):
            st.markdown(f"**{rank}. {name}** — `{example}`")
            st.caption(desc)

    with col2:
        st.subheader("🃏 How a Hand Works")
        st.markdown("""
**1. Blinds** — Small blind and big blind post forced bets.

**2. Hole cards** — You get 2 private cards. Don't show them.

**3. Pre-flop betting** — Fold, call, or raise based on your 2 cards.

**4. Flop** — 3 community cards appear. New betting round.

**5. Turn** — 4th community card. Bet again.

**6. River** — 5th and final card. Last chance to bet.

**7. Showdown** — Best 5-card hand (from your 2 + the 5 community) wins.
        """)

        st.subheader("🎯 Your Actions")
        st.markdown("""
| Action | When | Meaning |
|---|---|---|
| **Check** | Nobody has bet | Pass — stay in for free |
| **Call** | Someone bet | Match their bet |
| **Raise** | Anytime | Increase the bet |
| **Fold** | Anytime | Give up your hand |
        """)

    st.divider()
    st.subheader("💡 One Golden Rule")
    st.info(
        "Your **position** at the table matters. Acting **last** means you've seen everyone "
        "else's decisions before you make yours — a huge advantage. The **Button** (dealer chip) "
        "is the best position and moves clockwise each hand."
    )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ I'm ready — deal me in!", type="primary", use_container_width=True):
            ss.progress["tutorial_seen"] = True
            save_progress(ss.progress)
            ss.phase = "hand_start"
            st.rerun()
    with col_b:
        if st.button("⏭ Skip tutorial", use_container_width=True):
            ss.progress["tutorial_seen"] = True
            save_progress(ss.progress)
            ss.phase = "hand_start"
            st.rerun()


# ---------------------------------------------------------------------------
# Hand lifecycle helpers
# ---------------------------------------------------------------------------

def start_hand():
    game: GameState = ss.game
    ss.hand_number += 1
    ss.street_log = []
    ss.last_tip = None
    ss.show_tip = False
    ss.show_quant = False
    ss.last_feedback = None
    ss.hand_review_notes = []
    ss.quant_equity = None
    ss.quant_ev = None
    ss.quant_kelly = None
    ss.challenge_answer = ""
    ss.challenge_feedback = None

    dealer_idx = ss.dealer_index % len(game.players)
    game.start_hand(dealer_idx)

    # Quant mode: load concept for this hand
    if ss.mode == "quant":
        ss.current_concept = concept_for_hand(ss.hand_number)
        # If there's a pending challenge from last hand, keep it visible

    ss.current_bet = game.big_blind
    ss.phase = "betting"
    ss.action_queue = _build_action_queue(game.first_to_act_preflop())
    _advance_to_human_or_ai()


def _build_action_queue(first_index: int) -> list:
    n = len(ss.game.players)
    order = [(first_index + i) % n for i in range(n)]
    return [i for i in order if not ss.game.players[i].folded
            and not ss.game.players[i].is_all_in]


def _advance_to_human_or_ai():
    game: GameState = ss.game

    while ss.action_queue:
        idx = ss.action_queue[0]
        player = game.players[idx]

        if player.folded or player.is_all_in:
            ss.action_queue.pop(0)
            continue

        if idx == ss.human_index:
            to_call = ss.current_bet - player.bet_this_round
            ss.to_call = to_call
            ss.last_tip = pre_action_tip(
                player.hole_cards, game.community_cards,
                to_call, game.pot,
                ss.human_index, ss.dealer_index, len(game.players),
            )
            ss.show_tip = False

            # Quant mode: run Monte Carlo + EV + Kelly
            if ss.mode == "quant":
                num_opponents = len([p for p in game.players if not p.folded]) - 1
                eq = monte_carlo.run(
                    player.hole_cards, game.community_cards,
                    num_opponents=max(num_opponents, 1),
                    num_simulations=1000,
                )
                ss.quant_equity = eq
                min_raise = max(ss.current_bet * 2, game.big_blind)
                ss.quant_ev = ev_calculator.compute(
                    equity=eq["win_pct"],
                    pot=game.pot,
                    to_call=to_call,
                    raise_amount=min_raise,
                )
                ss.quant_kelly = kelly.kelly_fraction(
                    equity=eq["win_pct"],
                    pot=game.pot,
                    to_call=to_call,
                )
            return

        # AI acts
        to_call = ss.current_bet - player.bet_this_round
        action, amount = ai_decide(player, to_call, ss.current_bet,
                                   game.community_cards, game.pot)
        _apply_action(player, action, amount, game)
        game.history.record(player.name, game.street, action, amount,
                            player.hole_cards, game.community_cards)
        label = action.name + (f" ${amount}" if action == Action.RAISE else "")
        ss.street_log.append(f"**{player.name}**: {label}")

        if action == Action.RAISE:
            ss.current_bet = player.bet_this_round
            n = len(game.players)
            start = (idx + 1) % n
            remaining = [(start + i) % n for i in range(n - 1)]
            ss.action_queue = [i for i in remaining
                               if not game.players[i].folded
                               and not game.players[i].is_all_in
                               and i != idx]
        else:
            ss.action_queue.pop(0)

        active = [p for p in game.players if not p.folded]
        if len(active) == 1:
            ss.action_queue = []
            break

    _end_betting_round()


def _apply_action(player: Player, action: Action, amount: int, game: GameState):
    if action == Action.FOLD:
        player.folded = True
    elif action in (Action.CALL, Action.CHECK):
        to_call = ss.current_bet - player.bet_this_round
        if to_call > 0:
            game.pot += player.place_bet(to_call)
    elif action in (Action.RAISE, Action.ALL_IN):
        game.pot += player.place_bet(amount)


def _end_betting_round():
    game: GameState = ss.game
    active = [p for p in game.players if not p.folded]

    if len(active) == 1:
        ss.phase = "hand_over_no_showdown"
        return

    street = game.street
    if street == Street.PREFLOP:
        game.deal_flop()
        ss.street_log.append(f"🂠 **Flop:** {game.community_cards}")
        ss.current_bet = 0
        ss.action_queue = _build_action_queue(game.first_to_act_postflop())
        _advance_to_human_or_ai()
    elif street == Street.FLOP:
        game.deal_turn()
        ss.street_log.append(f"🂠 **Turn:** {game.community_cards[-1]}")
        ss.current_bet = 0
        ss.action_queue = _build_action_queue(game.first_to_act_postflop())
        _advance_to_human_or_ai()
    elif street == Street.TURN:
        game.deal_river()
        ss.street_log.append(f"🂠 **River:** {game.community_cards[-1]}")
        ss.current_bet = 0
        ss.action_queue = _build_action_queue(game.first_to_act_postflop())
        _advance_to_human_or_ai()
    elif street == Street.RIVER:
        ss.phase = "showdown"


# ---------------------------------------------------------------------------
# Human action handler
# ---------------------------------------------------------------------------

def handle_human_action(action: Action, raise_to: int = 0):
    game: GameState = ss.game
    player = game.players[ss.human_index]
    to_call = ss.to_call

    # Grade the action
    ss.last_feedback = evaluate_action(
        action, player.hole_cards, game.community_cards, to_call, game.pot
    )

    # Record for adaptive learning
    if ss.progress:
        record_decision(ss.progress, action, player.hole_cards,
                        game.community_cards, to_call, game.pot)

    if action == Action.FOLD:
        amount = 0
    elif action in (Action.CHECK, Action.CALL):
        amount = to_call
    else:
        amount = raise_to

    _apply_action(player, action, amount, game)
    game.history.record(player.name, game.street, action, amount,
                        player.hole_cards, game.community_cards)

    label = action.name
    if action == Action.RAISE:
        label += f" to ${player.bet_this_round}"
    ss.street_log.append(f"**You**: {label}")

    if action == Action.RAISE:
        ss.current_bet = player.bet_this_round
        n = len(game.players)
        idx = ss.human_index
        start = (idx + 1) % n
        remaining = [(start + i) % n for i in range(n - 1)]
        ss.action_queue = [i for i in remaining
                           if not game.players[i].folded
                           and not game.players[i].is_all_in]
    else:
        ss.action_queue.pop(0)

    active = [p for p in game.players if not p.folded]
    if len(active) == 1:
        ss.action_queue = []
        _end_betting_round()
        return

    _advance_to_human_or_ai()


# ---------------------------------------------------------------------------
# Coaching panel helper
# ---------------------------------------------------------------------------

def render_coaching_panel(tip: dict):
    """Display the structured coaching tip in labelled sections."""
    level = tip["tip_level"]
    icon = {"success": "✅", "warning": "⚠️", "info": "💡"}[level]
    qual_color = {
        "Premium": "🟢", "Strong": "🟢", "Playable": "🟡",
        "Marginal": "🟠", "Weak": "🔴"
    }.get(tip["hand_quality"], "⚪")

    with st.container(border=True):
        st.markdown(f"### {icon} Coaching Tip")

        m1, m2 = st.columns(2)
        m1.metric("Hand Quality", f"{qual_color} {tip['hand_quality']}")
        m2.metric("Win Chance", f"~{tip['equity_pct']}%")

        st.markdown("---")

        st.markdown(f"**🃏 Your Hand**")
        st.caption(tip["your_hand"] or "—")

        if tip.get("pot_odds") and tip["pot_odds_pct"] is not None:
            st.markdown(f"**💰 Pot Odds** ({tip['pot_odds_pct']}% to call)")
            st.caption(tip["pot_odds"])

        if tip.get("position_advice"):
            st.markdown(f"**📍 Position** — {tip['position']}")
            st.caption(tip["position_advice"])

        st.markdown("**🎯 Recommendation**")
        st.info(tip["recommendation"])


# ---------------------------------------------------------------------------
# Quant panel
# ---------------------------------------------------------------------------

def _build_game_context() -> dict:
    """Package current hand's quant numbers for the LLM."""
    eq = ss.quant_equity or {}
    ev = ss.quant_ev or {}
    kl = ss.quant_kelly or {}
    game = ss.game
    return {
        "pot": game.pot if game else 0,
        "to_call": ss.to_call,
        "equity": eq.get("win_pct", 0),
        "ev_call": ev.get("ev_call"),
        "ev_raise": ev.get("ev_raise"),
        "kelly_fraction": kl.get("full_kelly_fraction"),
    }


def render_quant_panel():
    """Full quant analysis panel — shown when player clicks the button in quant mode."""
    eq = ss.quant_equity
    ev = ss.quant_ev
    kl = ss.quant_kelly
    concept = ss.current_concept

    with st.container(border=True):
        st.markdown("### 🔬 Quant Analysis")

        # --- Equity (Monte Carlo) ---
        if eq:
            st.markdown("**Equity** *(Monte Carlo, n=1,000 simulations)*")
            c1, c2, c3 = st.columns(3)
            c1.metric("Win", f"{round(eq['win_pct']*100, 1)}%")
            c2.metric("Tie", f"{round(eq['tie_pct']*100, 1)}%")
            c3.metric("Lose", f"{round(eq['lose_pct']*100, 1)}%")
            st.caption(f"95% confidence interval: ±{round(eq['std_error']*100, 1)}%")
            st.divider()

        # --- EV at decision node ---
        if ev:
            st.markdown("**Expected Value at this decision node**")
            ev_fold = ev["ev_fold"]
            ev_call = ev["ev_call"]
            ev_raise = ev["ev_raise"]

            ev_cols = st.columns(3)
            ev_cols[0].metric("EV(Fold)", f"${ev_fold:.2f}")
            label = "EV(Call)" if ss.to_call > 0 else "EV(Check)"
            color_call = "normal" if ev_call >= 0 else "inverse"
            ev_cols[1].metric(label, f"${ev_call:.2f}",
                              delta="+" if ev_call > 0 else None)
            if ev_raise is not None:
                ev_cols[2].metric("EV(Raise)*", f"${ev_raise:.2f}",
                                  delta="+" if ev_raise > 0 else None)
            st.caption("*Raise EV assumes 30% fold equity — opponents fold 30% of the time to a raise.")
            best = ev.get("best_action", "")
            st.info(f"**Highest EV action: {best}**")
            st.divider()

        # --- Kelly Criterion ---
        if kl and kl.get("full_kelly_fraction") is not None:
            st.markdown("**Kelly Criterion — Optimal Bet Sizing**")
            k1, k2 = st.columns(2)
            k1.metric("Full Kelly fraction", f"{round(kl['full_kelly_fraction']*100, 1)}% of pot")
            k2.metric("½ Kelly (recommended)", f"${kl['half_kelly_bet']}")
            st.caption(kl["explanation"])
            st.divider()

        # --- Concept card for this hand ---
        if concept:
            with st.expander(f"📐 Theory this hand: {concept['title']}", expanded=True):
                st.latex(concept["formula"])
                st.markdown(concept["explanation"])
                st.markdown(f"**In trading:** {concept['trading']}")
                st.caption(f"📚 {concept['reference']}")

        # --- Challenge question (from previous hand's concept) ---
        if ss.pending_challenge:
            prev = ss.pending_challenge
            st.divider()
            st.markdown(f"### ✏️ Challenge: {prev['title']}")
            st.markdown(prev["challenge"])

            answer = st.text_area(
                "Your answer",
                value=ss.challenge_answer,
                key="challenge_input",
                placeholder="Show your working — formula, numbers, conclusion...",
                height=120,
            )

            if st.button("Submit answer to AI coach", type="primary"):
                ss.challenge_answer = answer
                with st.spinner("Evaluating your answer..."):
                    ss.challenge_feedback = llm_coach.evaluate_challenge(
                        concept=prev,
                        player_answer=answer,
                        game_context=_build_game_context(),
                    )
                st.rerun()

            if ss.challenge_feedback:
                st.markdown("**Coach feedback:**")
                st.success(ss.challenge_feedback)


# ---------------------------------------------------------------------------
# Phase: betting
# ---------------------------------------------------------------------------

def show_game():
    game: GameState = ss.game
    human = game.players[ss.human_index]

    render_sidebar(ss.progress)

    # Header
    col_title, col_chips = st.columns([3, 1])
    with col_title:
        st.title(f"Hand #{ss.hand_number}  —  {game.street.value}")
    with col_chips:
        st.metric("Your chips", f"${human.chips}")
        st.metric("Pot", f"${game.pot}")

    # Adaptive weakness banner
    if ss.progress:
        weakness = get_weakness_banner(ss.progress)
        if weakness:
            st.warning(f"**📌 Focus area:** {weakness['label']} — {weakness['tip']}")

    st.divider()

    # Community cards
    st.subheader("Community Cards")
    if game.community_cards:
        render_cards(game.community_cards)
    else:
        st.caption("_(none yet — pre-flop)_")

    st.divider()

    # Opponents
    st.subheader("Opponents")
    opp_cols = st.columns(len(game.players) - 1)
    opp_idx = 0
    for i, p in enumerate(game.players):
        if i == ss.human_index:
            continue
        with opp_cols[opp_idx]:
            status = "❌ Folded" if p.folded else ("💀 All-in" if p.is_all_in else "🟢 In")
            st.markdown(f"**{p.name}**")
            st.caption(f"${p.chips} chips  |  {status}")
            st.caption(f"Bet this round: ${p.bet_this_round}")
            render_cards(p.hole_cards, hidden=True)
        opp_idx += 1

    st.divider()

    # Your hand + action feedback
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Your Hand")
        render_cards(human.hole_cards)
        st.caption(f"Chips: ${human.chips}  |  Bet this round: ${human.bet_this_round}")

        if ss.last_feedback:
            fb = ss.last_feedback
            grade_icon = {"good": "✅", "ok": "🟡", "mistake": "❌"}[fb["grade"]]
            color = {"good": "success", "ok": "warning", "mistake": "error"}[fb["grade"]]
            getattr(st, color)(f"{grade_icon} **Last action:** {fb['explanation']}")

    with right:
        waiting_for_human = ss.action_queue and ss.action_queue[0] == ss.human_index
        if waiting_for_human:
            if ss.mode == "quant":
                if not ss.show_quant:
                    if st.button("🔬 Show quant analysis", use_container_width=True):
                        ss.show_quant = True
                        st.rerun()
                else:
                    render_quant_panel()
                    if st.button("🙈 Hide analysis", use_container_width=True):
                        ss.show_quant = False
                        st.rerun()
            elif ss.last_tip:
                if not ss.show_tip:
                    if st.button("💡 Show coaching tip", use_container_width=True):
                        ss.show_tip = True
                        st.rerun()
                else:
                    render_coaching_panel(ss.last_tip)
                    if st.button("🙈 Hide tip", use_container_width=True):
                        ss.show_tip = False
                        st.rerun()

    st.divider()

    # Action buttons
    waiting_for_human = ss.action_queue and ss.action_queue[0] == ss.human_index

    if waiting_for_human and not human.folded:
        to_call = ss.to_call
        st.subheader("Your Action")
        btn_cols = st.columns(4)

        with btn_cols[0]:
            if to_call == 0:
                if st.button("✅ Check", use_container_width=True):
                    handle_human_action(Action.CHECK)
                    st.rerun()
            else:
                if st.button(f"📞 Call ${to_call}", use_container_width=True):
                    handle_human_action(Action.CALL)
                    st.rerun()

        with btn_cols[1]:
            if st.button("❌ Fold", use_container_width=True):
                handle_human_action(Action.FOLD)
                st.rerun()

        with btn_cols[2]:
            min_raise = max(ss.current_bet * 2, game.big_blind)
            raise_amt = st.number_input(
                "Raise to", min_value=min_raise, max_value=human.chips,
                value=min(min_raise, human.chips), step=game.big_blind,
                key="raise_input",
            )

        with btn_cols[3]:
            if st.button("⬆ Raise", use_container_width=True, type="primary"):
                handle_human_action(Action.RAISE, raise_to=raise_amt)
                st.rerun()

    elif not waiting_for_human and ss.phase == "betting":
        st.info("⏳ Waiting for opponents...")
        st.rerun()

    if ss.street_log:
        with st.expander("Action log"):
            for entry in ss.street_log:
                st.markdown(entry)


# ---------------------------------------------------------------------------
# Shared post-hand rendering helpers
# ---------------------------------------------------------------------------

def _render_hand_review(game: GameState, human_name: str):
    st.subheader("📚 Hand Review — What could you improve?")
    notes = hand_review(game.history.decisions, human_name)
    for note in notes:
        icon = {"good": "✅", "mistake": "❌", "info": "💡"}[note["type"]]
        color = {"good": "success", "mistake": "error", "info": "info"}[note["type"]]
        getattr(st, color)(f"{icon} **{note['street']}:** {note['message']}")


def _render_chip_counts(players):
    st.subheader("Chip Counts")
    cols = st.columns(len(players))
    for i, p in enumerate(players):
        label = "You" if p.player_type == PlayerType.HUMAN else p.name
        cols[i].metric(label, f"${p.chips}")


def _advance_to_next_hand(game: GameState, human_won: bool):
    human = game.players[ss.human_index]
    if ss.progress:
        record_hand_result(ss.progress, won=human_won, chips_end=human.chips)

    # Quant mode: current concept becomes the pending challenge for next hand
    if ss.mode == "quant" and ss.current_concept:
        ss.pending_challenge = ss.current_concept
        ss.challenge_answer = ""
        ss.challenge_feedback = None

    ss.dealer_index = (ss.dealer_index + 1) % len(game.players)
    game.players = [p for p in game.players if p.chips > 0]

    if len(game.players) < 2 or human.chips <= 0:
        ss.phase = "game_over"
    else:
        for i, p in enumerate(game.players):
            if p.player_type == PlayerType.HUMAN:
                ss.human_index = i
                break
        ss.phase = "hand_start"
    st.rerun()


# ---------------------------------------------------------------------------
# Phase: showdown
# ---------------------------------------------------------------------------

def show_showdown():
    game: GameState = ss.game
    human = game.players[ss.human_index]

    render_sidebar(ss.progress)
    st.title(f"Hand #{ss.hand_number} — Showdown")
    st.divider()

    st.subheader("Community Cards")
    render_cards(game.community_cards)
    st.divider()

    st.subheader("Hands Revealed")
    active = [p for p in game.players if not p.folded]
    cols = st.columns(max(len(active), 1))
    for i, p in enumerate(active):
        score, five, hand_name = best_hand(p.hole_cards, game.community_cards)
        with cols[i]:
            label = "⭐ You" if p.player_type == PlayerType.HUMAN else p.name
            st.markdown(f"**{label}**")
            render_cards(p.hole_cards)
            st.caption(f"**{hand_name}**")
            render_cards(five)

    winners = game.showdown()
    game.award_pot(winners)
    human_won = human in winners

    st.divider()
    winner_names = ", ".join(
        "You" if w.player_type == PlayerType.HUMAN else w.name for w in winners
    )
    if human_won:
        st.success(f"🏆 You win the pot!")
    else:
        st.error(f"💸 {winner_names} wins this hand.")

    st.divider()
    _render_hand_review(game, human.name)

    if ss.mode == "quant":
        st.divider()
        st.subheader("🤖 AI Coach — Post-Hand Analysis")
        if not llm_coach.is_running():
            st.caption("Ollama is not running — start it to enable AI analysis.")
        else:
            if st.button("Ask AI coach to review this hand", use_container_width=True):
                weakness = get_weakness_banner(ss.progress) if ss.progress else None
                ctx = _build_game_context()
                ctx["pot"] = game.pot
                with st.spinner("Analysing hand..."):
                    analysis = llm_coach.post_hand_analysis(
                        hand_decisions=game.history.decisions,
                        player_name=human.name,
                        community_cards=game.community_cards,
                        weakness=weakness,
                        game_context=ctx,
                    )
                st.markdown(analysis)

    st.divider()
    _render_chip_counts(game.players)

    st.divider()
    if st.button("▶ Next Hand", type="primary", use_container_width=True):
        _advance_to_next_hand(game, human_won)


# ---------------------------------------------------------------------------
# Phase: hand over without showdown
# ---------------------------------------------------------------------------

def show_hand_over_no_showdown():
    game: GameState = ss.game
    human = game.players[ss.human_index]
    active = [p for p in game.players if not p.folded]
    winner = active[0]
    human_won = winner.player_type == PlayerType.HUMAN

    render_sidebar(ss.progress)
    st.title(f"Hand #{ss.hand_number} — Hand Over")
    st.divider()

    if human_won:
        st.success(f"🏆 Everyone folded — you win the pot of ${game.pot}!")
    else:
        st.error(f"💸 You folded. {winner.name} wins the pot of ${game.pot}.")

    game.award_pot([winner])

    st.divider()
    _render_hand_review(game, human.name)
    st.divider()
    _render_chip_counts(game.players)

    st.divider()
    if st.button("▶ Next Hand", type="primary", use_container_width=True):
        _advance_to_next_hand(game, human_won)


# ---------------------------------------------------------------------------
# Phase: game over
# ---------------------------------------------------------------------------

def show_game_over():
    render_sidebar(ss.progress)
    st.title("🏁 Game Over")

    human = next((p for p in ss.game.players if p.player_type == PlayerType.HUMAN), None)
    if human and human.chips > 0:
        st.success(f"You won the game with ${human.chips}! Well played.")
    else:
        st.error("You ran out of chips. Better luck next time!")

    if ss.progress:
        st.subheader("Your Session")
        col1, col2, col3 = st.columns(3)
        col1.metric("Hands Played", ss.progress["hands_played"])
        col2.metric("Hands Won", ss.progress["hands_won"])
        wr = win_rate(ss.progress)
        col3.metric("Win Rate", f"{wr}%" if wr else "—")

        weakness = get_weakness_banner(ss.progress)
        if weakness:
            st.info(f"**Keep working on:** {weakness['label']}\n\n{weakness['tip']}")

    st.divider()
    if st.button("Play Again", type="primary", use_container_width=True):
        import pathlib
        p = pathlib.Path("progress.json")
        if p.exists():
            p.unlink()
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()


# ---------------------------------------------------------------------------
# Phase: Kuhn Poker Lab
# ---------------------------------------------------------------------------

def show_kuhn_lab():
    with st.sidebar:
        st.title("🔬 Kuhn Poker Lab")
        st.markdown("A 3-card toy game that proves Nash Equilibrium exists in poker.")
        with st.expander("GTO Strategy (Nash Equilibrium)", expanded=True):
            st.markdown("""
**Player 1:**
- K → always Bet
- Q → always Check
- J → Bet with prob **1/3** (bluff), Check **2/3**

**Player 2 (facing a bet):**
- K → always Call
- Q → Call with prob **1/3**, Fold **2/3**
- J → always Fold

**Game value:** −1/18 chips/hand for P1.
""")
        with st.expander("Trading Analogy"):
            st.markdown("""
GTO = the strategy a rational market maker uses.
Exploiting a deviation = finding alpha against a mispricing counterparty.
The moment everyone plays GTO, no edge remains — like an efficient market.
""")
        if st.button("← Back to setup"):
            ss.phase = "setup"
            st.rerun()

    st.title("🔬 Kuhn Poker Lab — GTO vs Exploitative Play")
    st.markdown("Kuhn Poker strips poker to its mathematical core: 3 cards, 2 players, 1 decision each. The Nash Equilibrium is analytically solvable.")
    st.divider()

    # EV comparison table
    st.subheader("Strategy EV Comparison")
    if ss.kuhn_ev_summary is None:
        with st.spinner("Running 5,000 simulations per strategy..."):
            ss.kuhn_ev_summary = kuhn_poker.gto_vs_exploitative_summary()
    summary = ss.kuhn_ev_summary
    c1, c2, c3 = st.columns(3)
    c1.metric("GTO (Nash EQ) EV", f"{summary['ev_gto']:+.4f} chips/hand")
    c2.metric("Loose strategy EV", f"{summary['ev_loose']:+.4f} chips/hand")
    c3.metric("Cost of deviation", f"{summary['ev_diff']:+.4f} chips/hand")
    st.caption(summary["note"])
    st.divider()

    # Play a hand
    st.subheader("Play a Hand")
    if st.button("🃏 Deal new hand", use_container_width=False):
        ss.kuhn_hand = kuhn_poker.deal()
        ss.kuhn_step = "p1_act"
        ss.kuhn_p2_bet = False
        ss.kuhn_log = []
        st.rerun()

    hand = ss.kuhn_hand
    if hand:
        st.markdown(f"**Your card (Player 1):** `{hand.p1_card}` &nbsp;&nbsp; Opponent's card: `?`")
        st.caption(f"Pot: {hand.pot} chips (1 ante each). Bet size = 1 chip.")

        if ss.kuhn_step == "p1_act":
            st.markdown("**Your action:**")
            col_b, col_c = st.columns(2)
            gto_rec = kuhn_poker.GTO_P1[hand.p1_card]
            with col_b:
                st.caption(f"GTO: Bet {round(gto_rec['bet']*100)}%")
                if st.button("Bet (1 chip)", use_container_width=True):
                    hand.p1_action = "bet"
                    a2 = kuhn_poker.gto_action(kuhn_poker.GTO_P2_FACING_BET, hand.p2_card)
                    hand.p2_action = a2
                    hand.result = kuhn_poker.resolve(hand, "bet", a2)
                    ss.kuhn_log.append(f"You bet → Opponent {a2}s")
                    ss.kuhn_step = "done"
                    st.rerun()
            with col_c:
                st.caption(f"GTO: Check {round(gto_rec['check']*100)}%")
                if st.button("Check", use_container_width=True):
                    hand.p1_action = "check"
                    a2 = kuhn_poker.gto_action(kuhn_poker.GTO_P2_FACING_CHECK, hand.p2_card)
                    hand.p2_action = a2
                    ss.kuhn_log.append(f"You check → Opponent {a2}s")
                    if a2 == "bet":
                        ss.kuhn_step = "p1_respond"
                        ss.kuhn_p2_bet = True
                    else:
                        hand.result = kuhn_poker.resolve(hand, "check", "check")
                        ss.kuhn_step = "done"
                    st.rerun()

        elif ss.kuhn_step == "p1_respond":
            st.markdown("**Opponent bet 1 chip. Your response:**")
            col_ca, col_fo = st.columns(2)
            with col_ca:
                if st.button("Call", use_container_width=True):
                    hand.p1_response = "call"
                    hand.result = kuhn_poker.resolve(hand, "check", "bet", "call")
                    ss.kuhn_log.append("You call")
                    ss.kuhn_step = "done"
                    st.rerun()
            with col_fo:
                if st.button("Fold", use_container_width=True):
                    hand.p1_response = "fold"
                    hand.result = kuhn_poker.resolve(hand, "check", "bet", "fold")
                    ss.kuhn_log.append("You fold")
                    ss.kuhn_step = "done"
                    st.rerun()

        if ss.kuhn_step == "done" and hand.result:
            st.divider()
            result = hand.result
            if result["winner"] == "P1":
                st.success(f"🏆 You win! {result['desc']}  Profit: +{result['p1_profit']} chip(s)")
            else:
                st.error(f"💸 You lose. {result['desc']}  P&L: {result['p1_profit']} chip(s)")

            st.markdown(f"**Opponent's card was:** `{hand.p2_card}`")

            gto_p1 = kuhn_poker.GTO_P1[hand.p1_card]
            st.info(
                f"**GTO says:** Bet {round(gto_p1['bet']*100)}%, "
                f"Check {round(gto_p1['check']*100)}% with `{hand.p1_card}`. "
                + (f"With `{hand.p1_card}` facing a bet: "
                   f"Call {round(kuhn_poker.GTO_P2_FACING_BET[hand.p1_card]['call']*100)}%"
                   if ss.kuhn_p2_bet else "")
            )

        for entry in ss.kuhn_log:
            st.caption(f"› {entry}")

    st.divider()

    # Ask coach anything
    st.subheader("💬 Ask the Coach")
    question = st.text_input("Ask anything about Kuhn Poker, GTO, Nash Equilibrium, or the trading connection:",
                             key="kuhn_question")
    if st.button("Ask", key="kuhn_ask") and question.strip():
        with st.spinner("Thinking..."):
            answer = llm_coach.explain_concept(question)
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if ss.phase == "setup":
    show_setup()
elif ss.phase == "kuhn_lab":
    show_kuhn_lab()
elif ss.phase == "tutorial":
    render_sidebar(ss.progress)
    show_tutorial()
elif ss.phase == "hand_start":
    start_hand()
    st.rerun()
elif ss.phase == "betting":
    show_game()
elif ss.phase == "showdown":
    show_showdown()
elif ss.phase == "hand_over_no_showdown":
    show_hand_over_no_showdown()
elif ss.phase == "game_over":
    show_game_over()
