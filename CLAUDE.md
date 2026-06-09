# CLAUDE.md — Learn to Play Poker

Project context for Claude Code. Read this before making any changes.

## What this is

A Texas Hold'em poker learning game built in Python + Streamlit. Deployable to Streamlit Cloud for free. No paid APIs. This is a portfolio piece.

**Live app:** https://learn-to-play-poker.streamlit.app  
**Run locally:** `streamlit run app.py` → http://localhost:8501

## Two game modes

- **Standard** — poker strategy focus. Coaching tip on every decision, instant grading, post-hand review.
- **Quant Trading** — same game but each hand teaches a quant finance concept (EV, Kelly Criterion, Monte Carlo equity, pot odds as break-even analysis). Also includes a Kuhn Poker Lab for GTO theory.

## Stack

- `app.py` — entire UI and game state machine (~1850 lines). All Streamlit code lives here.
- `poker/` — pure Python game engine (no UI dependencies)
  - `game.py` — GameState, HandHistory, Street/Action enums
  - `player.py` — Player dataclass (chips, hole_cards, bet_this_round, folded, player_type)
  - `card.py` — Card, Deck
  - `ai.py` — rule-based AI: Tight / Loose / Aggressive
  - `hand_evaluator.py` — `best_hand(hole, community)` → (score, five_cards, hand_name)
  - `coaching.py` — `pre_action_tip()`, `evaluate_action()`, `hand_review()`
  - `progress.py` — JSON persistence, `save_progress()`, `load_progress()`, `win_rate()`, `get_weakness_banner()`
  - `monte_carlo.py` — equity simulation
  - `ev_calculator.py` — EV at each decision node
  - `kelly.py` — Kelly Criterion bet sizing
  - `quant_concepts.py` — per-hand concept cards
  - `llm_coach.py` — Ollama integration (optional, local only)
  - `kuhn_poker.py` — Kuhn Poker Lab

## app.py structure

Key functions in order:

| Function | Purpose |
|---|---|
| `init_state()` | Session state defaults. All state lives in `ss = st.session_state` |
| `render_sidebar()` | Progress stats + hand rankings reference |
| `_render_game_nav()` | Top bar with ⚙ Main Menu and 🔄 Restart buttons — shown on all game pages |
| `render_poker_table()` | iframe table via `components.html()`. Opponents overlap top of felt, human overlaps bottom — creates circular table feel. Coaching tip popup on hover. |
| `show_setup()` | Landing page. Mode cards are `st.button()` styled as cards (primary = selected). Continue screen shows if `ss.progress` is set. |
| `show_tutorial()` | One-page tutorial, single CTA button |
| `start_hand()` | Deals cards, posts blinds, snapshots `chips_at_hand_start` |
| `show_game()` | Main betting phase. Order: table → action buttons → strategy panel → quant toggle → ask coach |
| `handle_human_action()` | Processes fold/call/check/raise, records to history |
| `show_showdown()` | Reveals all hands, awards pot, shows hand snapshot + review |
| `show_hand_over_no_showdown()` | Everyone else folded, shows hand snapshot + review |
| `_render_hand_snapshot()` | Compact end-of-hand panel: board, your cards, chip deltas, per-street action badges |
| `_render_hand_review()` | Per-street expandable coaching notes |
| `_render_folded_hand()` | Shows what hand you would have made if you folded |

## Game state machine phases

`setup` → `tutorial` → `hand_start` → `betting` → `showdown` or `hand_over_no_showdown` → `game_over`  
(loops back to `hand_start` via Next Hand button)

Key session state keys:
- `ss.phase` — current phase
- `ss.game` — GameState object
- `ss.human_index` — index of human player in game.players
- `ss.dealer_index` — rotates each hand
- `ss.action_queue` — list of player indices still to act
- `ss.street_log` — list of action strings for current street
- `ss.last_tip` — coaching tip dict from `pre_action_tip()`
- `ss.mode` — "standard" or "quant" (active mode)
- `ss.setup_mode` — "standard" or "quant" (selected on setup page)
- `ss.progress` — progress dict (None = new user, truthy = returning user)
- `ss.chips_at_hand_start` — {player_name: chips} snapshot for computing deltas

## HTML rendering

Streamlit sanitizes HTML in `st.markdown()`. Anything with cards or the poker table uses `components.html()` which renders a full isolated iframe. This is why `import streamlit.components.v1 as components` is at the top.

## Styling

- App background: `#1c2030` (dark navy, injected via global CSS in `init_state()` area)
- Felt: radial green gradient, `border-radius: 50%`, wooden border
- Opponent seats overlap the TOP of the felt via `margin-bottom: -38px` + `z-index: 2`
- Human seat overlaps the BOTTOM via `margin-top: -38px` + `z-index: 2`
- Action badges: fold=red, raise=orange, call=green, check=blue

## Things that were tricky / non-obvious

- **HTML in iframes**: `st.markdown(unsafe_allow_html=True)` strips `<style>` and card spans render as raw text. Always use `components.html()` for anything with cards.
- **Continue screen logic**: uses `ss.progress` (session state), NOT `load_progress()` which reads from disk and would show "continue" to new users on Streamlit Cloud.
- **Raise input crash**: `StreamlitValueBelowMinError` when `human.chips < min_raise`. Fix: `min_raise = min(max(ss.current_bet * 2, game.big_blind), human.chips)`.
- **Chip delta**: `chips_at_hand_start` is snapshotted in `start_hand()` BEFORE `game.start_hand()` posts blinds, then compared at end-of-hand.

## Git / deploy

- Repo: https://github.com/JesJH/learn_to_play_poker
- Push only at large milestones, not after every small change.
- Streamlit Cloud auto-deploys from main branch within ~1 min of push.

## What's NOT here yet (potential next features)

- Mobile layout improvements (table clips on narrow screens)
- Multiplayer or online opponents
- Hand history persistence across sessions
- Sound effects or animations
