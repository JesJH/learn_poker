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
import streamlit.components.v1 as components
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
        "setup_mode": "standard",
        "chips_at_hand_start": {},   # {player_name: chips} snapshot before blinds
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
ss = st.session_state

# Global theme — slightly lighter than Streamlit's near-black default
st.markdown("""<style>
.stApp { background-color: #1c2030 !important; }
section[data-testid="stSidebar"] { background-color: #222538 !important; }
div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
    background-color: transparent;
}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Phase: setup
# ---------------------------------------------------------------------------



def show_setup():
    st.title("🃏 Learn to Play Poker")
    st.caption("Texas Hold'em · Coaching on every decision · Two modes to choose from")

    # Only show "continue" if this browser session has already started a game
    saved = ss.progress

    if saved:
        st.divider()
        wr = win_rate(saved)
        st.success(
            f"Welcome back, **{saved['player_name']}**!  "
            f"${saved['chips']} chips · {saved['hands_played']} hands played"
            + (f" · {wr}% win rate" if wr else "")
        )

        st.markdown("#### Switch mode?")
        st.markdown("""<style>
div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"]) button {
    white-space: pre-line !important;
    text-align: left !important;
    height: auto !important;
    min-height: 60px !important;
    padding: 10px 16px !important;
    line-height: 1.6 !important;
    font-size: 0.84rem !important;
}
</style>""", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            if st.button("🃏 Standard\nCoaching & strategy focus",
                         key="cont_mode_std", use_container_width=True,
                         type="primary" if ss.setup_mode == "standard" else "secondary"):
                ss.setup_mode = "standard"
                st.rerun()
        with mc2:
            if st.button("🔬 Quant Trading\nMath & EV focus",
                         key="cont_mode_quant", use_container_width=True,
                         type="primary" if ss.setup_mode == "quant" else "secondary"):
                ss.setup_mode = "quant"
                st.rerun()

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Continue", type="primary", use_container_width=True):
                ss.mode = ss.setup_mode
                _start_game(saved["player_name"], saved["chips"],
                            small_blind=10, progress=saved)
        with c2:
            if st.button("🔄 Start fresh", use_container_width=True):
                import pathlib
                p = pathlib.Path("progress.json")
                if p.exists():
                    p.unlink()
                ss.progress = None
                st.rerun()
        return

    st.divider()

    # --- Quick setup row ---
    c_name, c_chips, c_blind, c_tut = st.columns([2, 1.5, 1.5, 2])
    with c_name:
        player_name = st.text_input("Your name", value="Player", max_chars=20)
    with c_chips:
        chips = st.number_input("Starting chips", min_value=200, max_value=10000,
                                value=1000, step=100)
    with c_blind:
        small_blind = st.number_input("Small blind", min_value=5, value=10, step=5)
        st.caption(f"Big blind: ${small_blind * 2}")
    with c_tut:
        st.write("")
        show_tut = st.checkbox("Show tutorial first", value=True)

    st.divider()
    st.markdown("#### How do you want to play?")

    # CSS for mode-selector card buttons — safe: continue/fresh buttons are only shown
    # when saved progress exists, at which point this section never renders (early return above)
    st.markdown("""<style>
div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"]) button {
    white-space: pre-line !important;
    text-align: left !important;
    height: auto !important;
    min-height: 168px !important;
    padding: 18px 20px !important;
    line-height: 1.75 !important;
    font-size: 0.84rem !important;
}
</style>""", unsafe_allow_html=True)

    std_label = "🃏 Standard   ★ Recommended\n\nFocus on poker strategy — fold, call, or raise.\n\n· Coaching tip before every decision\n· Instant grading after each action\n· Post-hand review: what went well\n· Tracks your decision patterns over time"
    quant_label = "🔬 Quant Trading\n\nSame game, but every hand teaches a trading concept.\n\n· Expected Value: is this call profitable long-term?\n· Kelly Criterion: optimal bet sizing for your edge\n· Monte Carlo: simulated win % vs. pot odds\n· Challenge question each hand to apply the concept"

    c1, c2 = st.columns(2)
    with c1:
        if st.button(std_label, key="mode_std", use_container_width=True,
                     type="primary" if ss.setup_mode == "standard" else "secondary"):
            ss.setup_mode = "standard"
            st.rerun()
    with c2:
        if st.button(quant_label, key="mode_quant", use_container_width=True,
                     type="primary" if ss.setup_mode == "quant" else "secondary"):
            ss.setup_mode = "quant"
            st.rerun()

    st.divider()

    left, right = st.columns([3, 1])
    with left:
        st.caption("Opponents: **Alex** (Tight — plays strong hands only) · **Blake** (Loose — calls most things) · **Casey** (Aggressive — raises to pressure you)")
    with right:
        if st.button("▶ Start Game", type="primary", use_container_width=True):
            progress = _default_progress(player_name, chips)
            progress["tutorial_seen"] = not show_tut
            save_progress(progress)
            ss.mode = ss.setup_mode
            _start_game(player_name, chips, small_blind, progress)

    st.markdown("&nbsp;")
    if st.button("🔬 Open Kuhn Poker Lab (GTO Theory)", use_container_width=False):
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
    if st.button("▶ I'm ready — deal me in!", type="primary"):
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
    # Snapshot chips before blinds are posted
    ss.chips_at_hand_start = {p.name: p.chips for p in game.players}
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
# Poker table visual
# ---------------------------------------------------------------------------

RANK_VALUE = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

def _strategic_hand_notes(hole_cards, community_cards) -> list[str]:
    """Return bullet-point strategic observations about the current cards."""
    if len(hole_cards) != 2:
        return []
    c1, c2 = hole_cards
    r1, r2, s1, s2 = c1.rank, c2.rank, c1.suit, c2.suit
    v1 = RANK_VALUE.index(r1) if r1 in RANK_VALUE else 0
    v2 = RANK_VALUE.index(r2) if r2 in RANK_VALUE else 0
    suit_name = {"♠": "spades", "♥": "hearts", "♦": "diamonds", "♣": "clubs"}
    notes = []

    # Pocket pair
    if r1 == r2:
        notes.append(f"**Pocket {r1}s** — you already have a pair before the flop. Watch for a third {r1} on the board (trips), which would be very strong.")
    else:
        # High card strength
        high, low = (r1, r2) if v1 > v2 else (r2, r1)
        hv = max(v1, v2)
        if {r1, r2} == {'A', 'K'}:
            notes.append("**Ace-King (Big Slick)** — one of the best starting hands. Strong pair potential with top kicker on almost any board.")
        elif {r1, r2} == {'A', 'Q'} or {r1, r2} == {'A', 'J'}:
            notes.append(f"**Ace-{low}** — solid hand. An ace on the board gives you top pair with a strong kicker.")
        elif 'A' in [r1, r2]:
            notes.append(f"**Ace with a low kicker** — the ace is powerful but the {low} is weak. An ace on the board gives top pair, but be cautious if someone bets big — they may have a better kicker.")
        elif hv >= 10:
            notes.append(f"**Two high cards ({r1}, {r2})** — decent starting hand. Look for a pair on the flop using either card.")
        else:
            notes.append(f"**Low unconnected cards** — limited high-card strength. Hard to make top pair.")

        # Connectivity
        gap = abs(v1 - v2)
        if 1 <= gap <= 2:
            lo_v, hi_v = min(v1, v2), max(v1, v2)
            lo_r, hi_r = RANK_VALUE[lo_v], RANK_VALUE[hi_v]
            notes.append(f"**Connected cards** ({r1}-{r2}) — straight potential. Look for cards between {lo_r} and {hi_r} on the board to build a run of 5.")
        elif gap <= 4:
            notes.append(f"**Gapped cards** ({r1}-{r2}, gap of {gap}) — weak straight potential. Would need very specific board cards.")

    # Suited
    if s1 == s2:
        sn = suit_name.get(s1, s1)
        notes.append(f"**Suited** (both {sn}) — flush potential! If 3 more {sn} appear on the board across the flop/turn/river, you'll have a flush.")
    else:
        notes.append(f"**Off-suit** — no flush possible. Focus on pair and straight potential.")

    # Community card analysis
    if community_cards:
        board_ranks = [c.rank for c in community_cards]
        board_suits = [c.suit for c in community_cards]

        # Pair on board
        for hr in [r1, r2]:
            if hr in board_ranks:
                notes.append(f"**You paired the {hr}** on the board — you have a pair! The strength depends on your kicker and whether the board is dangerous.")
                break

        # Flush draw or flush
        if s1 == s2:
            matching = board_suits.count(s1)
            total_suited = 2 + matching
            sn = suit_name.get(s1, s1)
            if total_suited >= 5:
                notes.append(f"**You have a flush!** (5 {sn}) — very strong hand.")
            elif total_suited == 4:
                notes.append(f"**Flush draw** — one more {sn} on the board completes your flush.")
            elif total_suited == 3:
                notes.append(f"**Backdoor flush draw** — need 2 more {sn} cards. Unlikely but possible.")

        # Paired board (dangerous)
        from collections import Counter
        rank_counts = Counter(board_ranks)
        if any(v >= 2 for v in rank_counts.values()):
            notes.append("⚠️ **Board is paired** — someone could have three of a kind or a full house. Be careful if opponents bet big.")

    return notes


def _action_context(to_call: int, pot: int, human_chips: int, tip: dict | None) -> dict:
    """Return short situational captions for each action button."""
    equity_pct = tip["equity_pct"] if tip else None

    # Check / Call
    if to_call == 0:
        check_text = "Stay in for free — no chips at risk. Good when you want to see the next card cheaply."
    else:
        pot_after = pot + to_call
        break_even = round(to_call / pot_after * 100) if pot_after else 0
        if equity_pct is not None:
            ev_sign = "✓ profitable" if equity_pct > break_even else "✗ losing long-run"
            check_text = f"Need >{break_even}% equity to break even · You have ~{equity_pct}% → {ev_sign}"
        else:
            check_text = f"Need >{break_even}% equity to break even on this call."

    # Fold
    if to_call == 0:
        fold_text = f"Give up your cards. You'd be leaving a free check on the table — usually wrong."
    else:
        fold_text = f"Save ${to_call} but forfeit any claim to the ${pot} pot. Right when your hand is unlikely to win."

    # Raise
    raise_text = "Force opponents to put in more chips or fold. Best with a strong hand or as a calculated bluff."

    return {"call_check": check_text, "fold": fold_text, "raise": raise_text}


def _last_actions(street_log: list) -> dict:
    """Parse street_log and return {player_name: last_action_text}."""
    result = {}
    for entry in street_log:
        clean = entry.replace("**", "")
        if ":" in clean:
            name, action = clean.split(":", 1)
            result[name.strip()] = action.strip()
    return result


def _action_badge(action_text: str) -> str:
    """Return a coloured HTML badge for a player's last action."""
    t = action_text.upper()
    if "FOLD" in t:
        color, bg = "#ff5252", "rgba(255,82,82,0.15)"
    elif "RAISE" in t or "ALL_IN" in t:
        color, bg = "#ff9800", "rgba(255,152,0,0.15)"
    elif "CALL" in t:
        color, bg = "#4caf50", "rgba(76,175,80,0.15)"
    elif "CHECK" in t:
        color, bg = "#90caf9", "rgba(144,202,249,0.12)"
    else:
        color, bg = "#aaa", "rgba(170,170,170,0.1)"
    label = action_text.split()[0].capitalize() + (" " + " ".join(action_text.split()[1:]) if len(action_text.split()) > 1 else "")
    return f'<span style="display:inline-block;background:{bg};color:{color};border:1px solid {color};border-radius:4px;padding:1px 7px;font-size:11px;font-weight:600;margin-top:4px;">{label}</span>'


def _player_status_info(player):
    if player.folded:
        return "Folded", "#888888"
    elif player.is_all_in:
        return "All-in", "#ff6b6b"
    return "Active", "#4caf50"


def _table_card(card, hidden=False, large=False) -> str:
    size = "20px" if large else "15px"
    pad = "4px 10px" if large else "2px 7px"
    if hidden:
        return f'<span style="display:inline-block;background:#1a5276;color:white;border:2px solid #5a8aaa;border-radius:5px;padding:{pad};font-size:{size};margin:2px;min-width:30px;text-align:center;">🂠</span>'
    color = "#cc0000" if card.suit in ("♥", "♦") else "#111"
    return (
        f'<span style="display:inline-block;background:#ffffff;color:{color};'
        f'border:2px solid #bbb;border-radius:5px;padding:{pad};font-size:{size};'
        f'margin:2px;min-width:30px;text-align:center;font-weight:bold;'
        f'box-shadow:1px 1px 3px rgba(0,0,0,0.3);">'
        f'{card.rank}{card.suit}</span>'
    )


def render_poker_table(game, human, human_index: int, dealer_index: int, tip=None, last_actions=None, action_queue=None):
    players = game.players
    n = len(players)

    sb_index = (dealer_index + 1) % n
    bb_index = (dealer_index + 2) % n

    # Who acts next (first non-human in queue)
    next_to_act = None
    if action_queue:
        for idx in action_queue:
            if idx != human_index:
                next_to_act = idx
                break

    def _role_badge(idx):
        badges = []
        if idx % n == dealer_index % n:
            badges.append('<span style="background:#ffd700;color:#111;border-radius:3px;padding:1px 6px;font-size:9px;font-weight:bold;margin-right:2px;">Button</span>')
        if idx % n == sb_index:
            badges.append('<span style="background:#4fc3f7;color:#111;border-radius:3px;padding:1px 6px;font-size:9px;font-weight:bold;margin-right:2px;">Small Blind</span>')
        if idx % n == bb_index:
            badges.append('<span style="background:#ef9a9a;color:#111;border-radius:3px;padding:1px 6px;font-size:9px;font-weight:bold;margin-right:2px;">Big Blind</span>')
        return "".join(badges)

    # Action order: clockwise starting after dealer
    action_order = [(dealer_index + 1 + i) % n for i in range(n)]
    seat_order = {pid: pos + 1 for pos, pid in enumerate(action_order)}

    opp_seats = ""
    for i, p in enumerate(players):
        if i == human_index:
            continue
        cards = "".join(_table_card(c, hidden=True) for c in p.hole_cards)
        badges = _role_badge(i)
        short = p.name.split("(")[0].strip()
        order_num = seat_order.get(i, "")

        if p.folded:
            fold_overlay = '<div style="position:absolute;inset:0;background:rgba(0,0,0,0.55);border-radius:10px;display:flex;align-items:center;justify-content:center;z-index:2;"><span style="color:#ff5252;font-weight:bold;font-size:13px;letter-spacing:1px;">FOLDED</span></div>'
            border = "1px solid #3a1a1a"
            next_badge = ""
        else:
            fold_overlay = ""
            border = "2px solid #4caf50" if i == next_to_act else "1px solid #2a3050"
            next_badge = '<div style="font-size:9px;color:#4caf50;font-weight:bold;letter-spacing:1px;margin-bottom:2px;">▶ NEXT</div>' if i == next_to_act else ""

        last_act = (last_actions or {}).get(p.name, "")
        last_act_html = _action_badge(last_act) if last_act else '<span style="display:inline-block;min-height:20px;"></span>'

        bet_label = f'<div style="font-size:10px;color:#888;margin-top:2px;">This round: ${p.bet_this_round}</div>' if p.bet_this_round > 0 else '<div style="font-size:10px;color:transparent;">—</div>'

        opp_seats += f"""
        <div class="seat" style="border:{border};">
            {fold_overlay}
            <div>
                {next_badge}
                <div style="margin-bottom:2px;">{badges}</div>
                <div style="font-weight:bold;color:#ddd;font-size:13px;margin-bottom:2px;">{short}</div>
                <div style="margin:3px 0;">{cards}</div>
                <div style="font-size:12px;color:#ffd700;margin:2px 0;">💰 ${p.chips}</div>
                {bet_label}
            </div>
            <div style="margin-top:3px;">{last_act_html}</div>
        </div>"""

    if game.community_cards:
        comm = "".join(_table_card(c) for c in game.community_cards)
    else:
        comm = '<span style="color:rgba(255,255,255,0.5);font-size:13px;">Pre-flop — no cards yet</span>'

    h_status_text, h_status_color = _player_status_info(human)
    h_cards = "".join(_table_card(c, large=True) for c in human.hole_cards)
    h_role = _role_badge(human_index)
    h_order = seat_order.get(human_index, "")

    if human.folded:
        h_fold_overlay = '<div style="position:absolute;inset:0;background:rgba(0,0,0,0.55);border-radius:10px;display:flex;align-items:center;justify-content:center;z-index:2;"><span style="color:#ff5252;font-weight:bold;font-size:14px;letter-spacing:1px;">FOLDED</span></div>'
        h_border = "2px solid #3a1a1a"
    else:
        h_fold_overlay = ""
        h_border = "2px solid #ffd700"

    h_bet_label = f'<div style="font-size:11px;color:#aaa;">This round: ${human.bet_this_round}</div>' if human.bet_this_round > 0 else ""

    # Build coaching tip HTML
    tip_html = ""
    if tip:
        icon = {"success": "✅", "warning": "⚠️", "info": "💡"}.get(tip["tip_level"], "💡")
        qcolor = {"Premium": "#4caf50", "Strong": "#4caf50", "Playable": "#ffd700",
                  "Marginal": "#ff9800", "Weak": "#f44336"}.get(tip["hand_quality"], "#aaa")
        rec = tip.get("recommendation", "").replace('"', "&quot;").replace("<", "&lt;")
        extras = ""
        if tip.get("your_hand"):
            extras += f'<div style="font-size:12px;color:#ccc;margin-bottom:4px;"><b>Your hand:</b> {tip["your_hand"]}</div>'
        if tip.get("pot_odds"):
            extras += f'<div style="font-size:12px;color:#ccc;margin-bottom:4px;"><b>Pot odds:</b> {tip["pot_odds"]}</div>'
        if tip.get("position_advice"):
            extras += f'<div style="font-size:12px;color:#ccc;margin-bottom:4px;"><b>Position:</b> {tip["position_advice"]}</div>'

        tip_html = f"""
<div style="display:flex;justify-content:center;margin-top:10px;">
  <div class="tip-wrap">
    <div class="tip-trigger">{icon} Hover for coaching tip</div>
    <div class="tip-popup">
      <div style="font-weight:bold;color:#fff;font-size:14px;margin-bottom:8px;">{icon} Coaching Tip</div>
      <div style="font-size:12px;color:{qcolor};margin-bottom:6px;">Hand: <b>{tip['hand_quality']}</b> · Win: ~{tip['equity_pct']}%</div>
      {extras}
      <div style="font-size:12px;color:#90caf9;font-style:italic;margin-top:8px;border-top:1px solid #2a3050;padding-top:8px;">💬 {rec}</div>
    </div>
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 8px 6px 8px; overflow: hidden; }}
/* Opponents sit at the TOP EDGE of the felt — overlap pulls felt up under them */
.opps {{
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: nowrap;
    align-items: flex-end;
    position: relative;
    z-index: 2;
    margin-bottom: -38px;
    padding: 0 8px;
}}
.felt {{
    background: radial-gradient(ellipse at 50% 40%, #1e7a33 0%, #0f5220 55%, #072e10 100%);
    border: 7px solid #5a3510;
    border-radius: 50%;
    min-height: 160px;
    padding: 52px 60px 46px;
    margin: 0 auto;
    max-width: 92%;
    text-align: center;
    box-shadow: 0 8px 28px rgba(0,0,0,0.8), inset 0 0 50px rgba(0,0,0,0.35);
    position: relative;
    z-index: 1;
}}
/* Human sits at the BOTTOM EDGE — overlap pulls seat up over felt border */
.human-row {{
    display: flex;
    justify-content: center;
    position: relative;
    z-index: 2;
    margin-top: -38px;
}}
.human-seat {{
    background: rgba(10,28,10,0.97);
    border: {h_border};
    border-radius: 12px;
    padding: 8px 18px 10px;
    text-align: center;
    min-width: 160px;
    position: relative;
}}
.seat {{
    background: rgba(15,20,42,0.97);
    border-radius: 10px;
    padding: 7px 10px;
    text-align: center;
    width: 138px;
    min-height: 148px;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}}
.tip-wrap {{ position: relative; display: inline-block; }}
.tip-trigger {{
    background: #1565c0; color: #fff; border-radius: 8px;
    padding: 8px 18px; font-size: 13px; cursor: default; user-select: none;
}}
.tip-popup {{
    display: none;
    position: absolute;
    bottom: 110%;
    left: 50%;
    transform: translateX(-50%);
    width: 320px;
    background: #1a1f2e;
    border: 1px solid #3a4060;
    border-radius: 10px;
    padding: 14px;
    z-index: 100;
    box-shadow: 0 -4px 24px rgba(0,0,0,0.7);
    text-align: left;
}}
.tip-wrap:hover .tip-popup {{ display: block; }}
</style>
</head><body>
<div class="opps">{opp_seats}</div>
<div class="felt">
    <div style="color:rgba(255,255,255,0.4);font-size:10px;text-transform:uppercase;letter-spacing:2px;margin-bottom:5px;">
        {game.street.value} — Community Cards
    </div>
    <div style="margin:4px 0;">{comm}</div>
    <div style="color:#ffd700;font-size:14px;font-weight:bold;margin-top:8px;">💰 Pot: ${game.pot}</div>
</div>
<div class="human-row">
    <div class="human-seat">
        {h_fold_overlay}
        <div style="margin-bottom:2px;">{h_role}</div>
        <div style="color:#ffd700;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:1px;">You</div>
        <div style="font-weight:bold;color:#eee;font-size:13px;margin-bottom:3px;">{human.name}</div>
        <div style="margin:4px 0;">{h_cards}</div>
        <div style="font-size:12px;color:#ffd700;margin:2px 0;font-weight:bold;">💰 ${human.chips}</div>
        {h_bet_label}
        <div style="font-size:11px;margin-top:2px;color:{h_status_color};font-weight:bold;">● {h_status_text}</div>
    </div>
</div>
{tip_html}
</body></html>"""

    components.html(html, height=490, scrolling=False)


# ---------------------------------------------------------------------------
# In-game navigation bar (main menu + restart)
# ---------------------------------------------------------------------------

def _render_game_nav():
    """Compact top-of-page navigation strip shown during all game phases."""
    left, mid, right = st.columns([3, 1, 1])
    with left:
        mode_label = "🔬 Quant Trading" if ss.mode == "quant" else "🃏 Standard"
        st.caption(f"Hand #{ss.hand_number} · Mode: **{mode_label}**")
    with mid:
        if st.button("⚙ Main Menu", use_container_width=True):
            ss.phase = "setup"
            st.rerun()
    with right:
        if st.button("🔄 Restart", use_container_width=True):
            # Keep name, chips start, small blind — wipe history and chips back to 1000
            p = ss.progress
            start_chips = 1000
            player_name = p["player_name"] if p else "Player"
            small_blind = ss.game.small_blind if ss.game else 10
            import pathlib
            path = pathlib.Path("progress.json")
            if path.exists():
                path.unlink()
            fresh = _default_progress(player_name, start_chips)
            save_progress(fresh)
            ss.progress = fresh
            ss.mode = ss.setup_mode
            _start_game(player_name, start_chips, small_blind, fresh)


# ---------------------------------------------------------------------------
# Phase: betting
# ---------------------------------------------------------------------------

def show_game():
    game: GameState = ss.game
    human = game.players[ss.human_index]

    render_sidebar(ss.progress)
    _render_game_nav()
    st.title(f"Hand #{ss.hand_number}  —  {game.street.value}")

    if ss.progress:
        weakness = get_weakness_banner(ss.progress)
        if weakness:
            st.warning(f"**📌 Focus area:** {weakness['label']} — {weakness['tip']}")

    waiting_for_human = ss.action_queue and ss.action_queue[0] == ss.human_index
    tip = ss.last_tip if (waiting_for_human and not human.folded) else None
    render_poker_table(game, human, ss.human_index, ss.dealer_index,
                       tip=tip, last_actions=_last_actions(ss.street_log),
                       action_queue=ss.action_queue)

    if ss.last_feedback:
        fb = ss.last_feedback
        grade_icon = {"good": "✅", "ok": "🟡", "mistake": "❌"}[fb["grade"]]
        color = {"good": "success", "ok": "warning", "mistake": "error"}[fb["grade"]]
        getattr(st, color)(f"{grade_icon} **Last action:** {fb['explanation']}")

    if waiting_for_human and not human.folded:
        to_call = ss.to_call
        ctx = _action_context(to_call, game.pot, human.chips, ss.last_tip)
        tip = ss.last_tip

        # --- Action buttons first, right under the table ---
        btn_cols = st.columns([2, 2, 3, 2])
        with btn_cols[0]:
            if to_call == 0:
                if st.button("✅ Check", use_container_width=True):
                    handle_human_action(Action.CHECK)
                    st.rerun()
            else:
                if st.button(f"📞 Call  ${to_call}", use_container_width=True, type="primary"):
                    handle_human_action(Action.CALL)
                    st.rerun()
            st.caption(ctx["call_check"])

        with btn_cols[1]:
            if st.button("✖ Fold", use_container_width=True):
                handle_human_action(Action.FOLD)
                st.rerun()
            st.caption(ctx["fold"])

        with btn_cols[2]:
            min_raise = min(max(ss.current_bet * 2, game.big_blind), human.chips)
            raise_amt = st.number_input(
                "Raise to", min_value=min_raise, max_value=human.chips,
                value=min_raise, step=game.big_blind,
                key="raise_input", label_visibility="collapsed",
            )

        with btn_cols[3]:
            if st.button("⬆ Raise", use_container_width=True, type="primary"):
                handle_human_action(Action.RAISE, raise_to=raise_amt)
                st.rerun()
            st.caption(ctx["raise"])

        # --- Strategy panel below actions ---
        hand_notes = _strategic_hand_notes(human.hole_cards, game.community_cards)
        is_weak = tip and tip.get("hand_quality") in ("Weak", "Marginal")

        with st.container(border=True):
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**🔍 Your Hand**")
                for note in hand_notes[:3]:
                    st.markdown(f"<span style='font-size:0.85rem;'>• {note}</span>", unsafe_allow_html=True)
            with right:
                if to_call > 0 and tip:
                    pot_after = game.pot + to_call
                    break_even = round(to_call / pot_after * 100) if pot_after else 0
                    eq = tip["equity_pct"]
                    ev_color = "#4caf50" if eq > break_even else "#f44336"
                    ev_label = "Profitable call ✓" if eq > break_even else "Losing call ✗"
                    be_formula = f"${to_call} ÷ (${game.pot} + ${to_call}) = {break_even}%"
                    eq_formula = "Estimated via Monte Carlo: simulates 1 000 random boards and counts how often your hand wins"
                    st.markdown("**💰 Pot Odds vs Your Equity**")
                    st.markdown(
                        f"<style>.calc-tip{{border-bottom:1px dashed #888;cursor:help;}}</style>"
                        f"<div style='font-size:1.05rem;font-weight:bold;'>"
                        f"Need <span class='calc-tip' title='Break-even %: {be_formula}'><b>{break_even}%</b></span>"
                        f" &nbsp;·&nbsp; "
                        f"You have <span class='calc-tip' title='Your equity: {eq_formula}' "
                        f"style='color:{ev_color};font-size:1.1rem;'><b>{eq}%</b></span>"
                        f"</div>"
                        f"<div style='color:{ev_color};font-size:0.9rem;font-weight:600;margin-top:2px;'>{ev_label}</div>",
                        unsafe_allow_html=True,
                    )
                elif to_call == 0:
                    st.markdown("**💡 Free action**")
                    st.caption("No bet to call — you can check and see the next card at no cost.")
                if is_weak and tip:
                    st.markdown("**⚠️ Recommendation**")
                    st.caption(tip.get("recommendation", "—"))

    elif not waiting_for_human and ss.phase == "betting":
        st.info("⏳ Waiting for opponents...")
        st.rerun()

    if waiting_for_human and ss.mode == "quant":
        if not ss.show_quant:
            if st.button("🔬 Show quant analysis", use_container_width=True):
                ss.show_quant = True
                st.rerun()
        else:
            render_quant_panel()
            if st.button("🙈 Hide analysis", use_container_width=True):
                ss.show_quant = False
                st.rerun()

    with st.expander("💬 Ask the Coach"):
        if not llm_coach.is_running():
            st.caption("Ollama is not running — start it locally to enable the AI coach.")
        else:
            question = st.text_input(
                "Ask anything about this hand, your cards, or poker strategy:",
                key="game_coach_question",
            )
            if st.button("Ask", key="game_coach_ask") and question.strip():
                with st.spinner("Thinking..."):
                    ss.ask_coach_answer = llm_coach.explain_concept(question)
            if ss.ask_coach_answer:
                st.markdown(ss.ask_coach_answer)


# ---------------------------------------------------------------------------
# Shared post-hand rendering helpers
# ---------------------------------------------------------------------------

def _render_hand_review(game: GameState, human_name: str):
    st.subheader("📚 Hand Review")
    notes = hand_review(game.history.decisions, human_name)

    if not notes:
        st.info("💡 No decisions recorded this hand.")
        return

    mistakes = [n for n in notes if n["assessment"] == "mistake"]
    goods = [n for n in notes if n["assessment"] == "good"]

    if mistakes:
        st.error(f"⚠️ {len(mistakes)} decision(s) to work on across {len(notes)} street(s).")
    elif goods:
        st.success(f"✅ Solid hand — {len(goods)} strong decision(s) out of {len(notes)}.")
    else:
        st.warning("🟡 Decent play — no big mistakes, but room to optimize.")

    for note in notes:
        a = note["assessment"]
        icon = {"good": "✅", "ok": "🟡", "mistake": "❌"}.get(a, "💡")
        label = f"{icon} {note['street']} — {note['action']}  ({note['hand_name']}, ~{note['equity_pct']}% equity)"
        with st.expander(label, expanded=(a == "mistake")):
            if note.get("went_well"):
                st.markdown(f"**✅ What went well:** {note['went_well']}")
            if note.get("risk"):
                st.markdown(f"**⚠️ Risk:** {note['risk']}")
            if note.get("ev_note"):
                st.markdown(f"**📊 EV context:** {note['ev_note']}")


def _render_folded_hand(game: GameState, human):
    """Show the human the hand they folded — what they had and what it would have made."""
    if not human.folded or not human.hole_cards:
        return
    st.divider()
    st.subheader("🃏 Your Folded Hand")
    st.caption("You folded — here's what you were holding and what you would have made:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your hole cards (folded)**")
        render_cards(human.hole_cards)
    if game.community_cards:
        score, five, hand_name = best_hand(human.hole_cards, game.community_cards)
        with col2:
            st.markdown(f"**Would have made: {hand_name}**")
            render_cards(five)
    else:
        with col2:
            st.caption("Hand ended before community cards — no board to evaluate against.")


def _render_chip_counts(players):
    st.subheader("Chip Counts")
    cols = st.columns(len(players))
    for i, p in enumerate(players):
        label = "You" if p.player_type == PlayerType.HUMAN else p.name
        cols[i].metric(label, f"${p.chips}")


def _render_hand_snapshot(game: GameState, human, pot_at_end: int):
    """Compact hand summary: board, your cards, chips, per-street actions."""
    from poker.hand_evaluator import best_hand

    # Build per-street action rows from history
    streets_order = ["preflop", "flop", "turn", "river"]
    street_actions: dict[str, list[tuple[str, str, int]]] = {}
    for d in game.history.decisions:
        s = d["street"]
        if s not in street_actions:
            street_actions[s] = []
        street_actions[s].append((d["player"], d["action"], d["amount"]))

    # Card HTML helpers (small inline cards)
    suit_color = {"♠": "#e0e0e0", "♥": "#ff6b6b", "♦": "#ff6b6b", "♣": "#e0e0e0"}

    def card_html(card, hidden=False):
        if hidden:
            return '<span style="display:inline-block;background:#2a2d3e;border:1px solid #555;border-radius:4px;padding:2px 6px;font-size:13px;margin:1px;">🂠</span>'
        s = str(card)[-1]
        color = suit_color.get(s, "#e0e0e0")
        return (f'<span style="display:inline-block;background:#1e2030;border:1px solid #444;'
                f'border-radius:4px;padding:2px 7px;font-size:13px;font-weight:700;color:{color};margin:1px;">{card}</span>')

    def badge(action: str, amount: int) -> str:
        a = action.upper()
        if "FOLD" in a:   c, bg = "#ff5252", "rgba(255,82,82,0.15)"
        elif "RAISE" in a or "ALL_IN" in a: c, bg = "#ff9800", "rgba(255,152,0,0.15)"
        elif "CALL" in a: c, bg = "#4caf50", "rgba(76,175,80,0.15)"
        elif "CHECK" in a: c, bg = "#90caf9", "rgba(144,202,249,0.12)"
        else:              c, bg = "#aaa", "rgba(170,170,170,0.1)"
        label = action.capitalize() + (f" ${amount}" if amount > 0 and "FOLD" not in a and "CHECK" not in a else "")
        return (f'<span style="background:{bg};color:{c};border:1px solid {c};border-radius:4px;'
                f'padding:1px 7px;font-size:11px;font-weight:600;">{label}</span>')

    board_html = "".join(card_html(c) for c in game.community_cards) if game.community_cards else "<span style='color:#666;font-size:12px;'>No community cards</span>"
    hole_html = "".join(card_html(c) for c in human.hole_cards) if human.hole_cards else ""

    hand_label = ""
    if human.hole_cards and game.community_cards:
        try:
            _, _, hand_label = best_hand(human.hole_cards, game.community_cards)
        except Exception:
            pass

    # Build player list with short names + chip delta
    chips_start = ss.get("chips_at_hand_start", {})
    player_rows = []
    for p in game.players:
        is_human = p.player_type == PlayerType.HUMAN
        short = "You" if is_human else p.name.split(" ")[0]
        name_color = "#4caf50" if is_human else "#aaa"
        chip_color = "#ffffff" if is_human else "#e0e0e0"
        border = "border:1px solid #4caf50;border-radius:8px;padding:6px 10px;" if is_human else "padding:6px 10px;"
        delta = p.chips - chips_start.get(p.name, p.chips)
        if delta > 0:
            delta_html = f'<div style="font-size:10px;color:#4caf50;margin-top:1px;">+${delta}</div>'
        elif delta < 0:
            delta_html = f'<div style="font-size:10px;color:#ff5252;margin-top:1px;">${delta}</div>'
        else:
            delta_html = f'<div style="font-size:10px;color:#666;margin-top:1px;">—</div>'
        player_rows.append(
            f'<div style="text-align:center;min-width:80px;{border}">'
            f'<div style="font-size:11px;color:{name_color};font-weight:{"700" if is_human else "400"};margin-bottom:2px;">{short}</div>'
            f'<div style="font-size:14px;font-weight:700;color:{chip_color};">${p.chips}</div>'
            f'{delta_html}'
            f'</div>'
        )

    # Build action table rows
    action_rows_html = ""
    for s in streets_order:
        if s not in street_actions:
            continue
        acts = street_actions[s]
        cells = ""
        for player_name, action, amount in acts:
            short = "You" if player_name == human.name else player_name.split(" ")[0]
            cells += (f'<td style="padding:3px 8px;white-space:nowrap;">'
                      f'<span style="font-size:10px;color:#888;margin-right:4px;">{short}</span>'
                      f'{badge(action, amount)}</td>')
        action_rows_html += (f'<tr><td style="padding:3px 8px;font-size:11px;font-weight:600;'
                             f'color:#aaa;white-space:nowrap;text-transform:capitalize;">{s}</td>'
                             f'{cells}</tr>')

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
body {{ background: transparent; color: #e0e0e0; }}
.snapshot {{ border: 1px solid #2a2d3e; border-radius: 10px; overflow: hidden; }}
.row {{ display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-bottom: 1px solid #2a2d3e; flex-wrap: wrap; }}
.row:last-child {{ border-bottom: none; }}
.label {{ font-size: 11px; font-weight: 600; color: #888; min-width: 52px; }}
.hand-name {{ font-size: 11px; color: #4caf50; margin-left: 8px; }}
.chips-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
table {{ border-collapse: collapse; width: 100%; }}
td {{ vertical-align: middle; }}
</style></head><body>
<div class="snapshot">
  <div class="row">
    <span class="label">Board</span>
    <span>{board_html}</span>
    <span style="margin-left:auto;font-size:12px;color:#aaa;">Pot&nbsp;<b style="color:#e0e0e0;">${pot_at_end}</b></span>
  </div>
  <div class="row">
    <span class="label">Your hand</span>
    <span>{hole_html}</span>
    {f'<span class="hand-name">— {hand_label}</span>' if hand_label else ''}
  </div>
  <div class="row">
    <span class="label">Chips</span>
    <div class="chips-row">{''.join(player_rows)}</div>
  </div>
  <div class="row" style="padding:0;">
    <table>{action_rows_html}</table>
  </div>
</div>
</body></html>"""

    n_streets = len([s for s in streets_order if s in street_actions])
    height = 190 + n_streets * 30
    components.html(html, height=height)


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
    _render_game_nav()
    st.title(f"Hand #{ss.hand_number} — Showdown")

    winners = game.showdown()
    pot_before = game.pot
    game.award_pot(winners)
    human_won = human in winners

    winner_names = ", ".join(
        "You" if w.player_type == PlayerType.HUMAN else w.name for w in winners
    )
    if human_won:
        st.success(f"🏆 You win the pot of ${pot_before}!")
    else:
        st.error(f"💸 {winner_names} wins the pot of ${pot_before}.")

    st.divider()
    _render_hand_snapshot(game, human, pot_before)

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
                ctx["pot"] = pot_before
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
    _render_game_nav()
    st.title(f"Hand #{ss.hand_number} — Hand Over")

    pot_before = game.pot
    game.award_pot([winner])

    if human_won:
        st.success(f"🏆 Everyone folded — you win the pot of ${pot_before}!")
    else:
        st.error(f"💸 You folded. {winner.name} wins the pot of ${pot_before}.")

    st.divider()
    _render_hand_snapshot(game, human, pot_before)

    _render_folded_hand(game, human)

    st.divider()
    _render_hand_review(game, human.name)

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
