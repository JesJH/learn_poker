# Design Document

Internal reference for how the game is structured and why.

---

## Architecture overview

The app is split into two layers:

```
┌──────────────────────────────────────────────────┐
│  app.py  —  Streamlit UI + state machine         │
│  (phases: setup → tutorial → betting → showdown) │
└────────────────────┬─────────────────────────────┘
                     │ calls
┌────────────────────▼─────────────────────────────┐
│  poker/  —  game engine (pure Python, no UI)     │
│  card · player · game · ai · hand_evaluator      │
│  coaching · progress                             │
└──────────────────────────────────────────────────┘
```

The game engine has no Streamlit imports — it can be tested independently or swapped to a different UI.

---

## Game engine modules

### `poker/card.py`
Defines `Card` (rank, suit, value) and `Deck` (shuffle + deal). Cards are compared by `.value` (0–12, where 0=2 and 12=A).

### `poker/player.py`
`Player` dataclass holds chips, hole cards, and betting state. Key methods:
- `place_bet(amount)` — deducts chips, caps at stack, sets `is_all_in` flag
- `reset_for_hand()` / `reset_for_round()` — cleans state between hands/streets
- `is_active` property — `True` unless folded or all-in

`PlayerType` enum: `HUMAN`, `AI_TIGHT`, `AI_LOOSE`, `AI_AGGRESSIVE`

### `poker/hand_evaluator.py`
`best_hand(hole_cards, community_cards)` — evaluates all C(7,5)=21 combinations and returns the best 5-card score, the cards that make it, and the hand name.

Hand scores are tuples — Python tuple comparison handles tiebreaking automatically (e.g. `(1, 12, 8, 3)` beats `(1, 12, 7, 5)` for One Pair Aces).

`hand_strength()` returns a 0–1 estimate based on hand rank, used for coaching hints (not game decisions).

### `poker/ai.py`
Three rule-based personalities:

| Style | Logic |
|---|---|
| Tight | Plays top ~30% of hands; raises only with strength ≥ 0.70 |
| Loose | Calls most hands; rarely folds unless grossly outmatched |
| Aggressive | Raises frequently; 15% bluff rate regardless of hand strength |

Strength is estimated via `_preflop_strength()` (pre-flop) or `hand_strength()` (post-flop).

### `poker/game.py`
`GameState` manages one hand. It is **not** a loop — each step is called explicitly by `app.py`:
1. `start_hand(dealer_index)` — shuffle, deal, post blinds
2. `run_betting_round(first_to_act, current_bet)` — NOT used by the app (replaced by the queue system)
3. `deal_flop()` / `deal_turn()` / `deal_river()` — burn + deal community cards
4. `showdown()` — returns list of winners
5. `award_pot(winners)` — distributes pot

`HandHistory` records every decision for post-hand review.

---

## Streamlit state machine

Streamlit reruns the entire `app.py` script on every user interaction. Game state is stored in `st.session_state` (`ss`). The phase variable drives which screen renders:

```
setup
  └─► tutorial (if first-time)
        └─► hand_start
              └─► betting ◄──────┐
                    ├─► showdown  │ (next hand)
                    └─► hand_over_no_showdown
```

### Why a queue instead of a loop

`game.run_betting_round()` is a blocking loop — incompatible with Streamlit's rerun model. Instead, `app.py` maintains `ss.action_queue`: a list of player indices still to act this round.

Each rerun:
1. Peek at `action_queue[0]`
2. If it's an AI → act, pop, loop
3. If it's human → compute coaching tip, stop and wait for button click
4. Queue empty → call `_end_betting_round()` to advance the street

---

## Coaching system

### Pre-action tip (`coaching.pre_action_tip`)

Called when it's the human's turn. Returns a structured dict with four labelled sections:

| Section | Content |
|---|---|
| `your_hand` | Hand quality and what to do with it |
| `pot_odds` | Whether the call cost is justified by equity (only shown when there's a bet to call) |
| `position_advice` | What acting early/late means right now |
| `recommendation` | One clear action sentence |

Tips are hidden by default — the player clicks "💡 Show coaching tip" to reveal. This forces the player to think first.

### Post-action feedback (`coaching.evaluate_action`)

Grades the decision immediately after it's made:
- **good** — mathematically sound
- **ok** — acceptable but leaves value on the table
- **mistake** — loses money long-term

### Post-hand review (`coaching.hand_review`)

Replays every human decision from the hand and flags notable moments (mistakes and notably good plays).

---

## Adaptive learning system (`poker/progress.py`)

Four decision patterns are counted every time the human acts:

| Pattern | Trigger condition |
|---|---|
| `preflop_loose` | Called/raised pre-flop with a weak or connector hand |
| `preflop_tight` | Folded pre-flop with a premium or strong hand |
| `ignored_pot_odds` | Called when equity < pot_odds_pct by more than 10 points |
| `too_passive` | Checked or called with equity ≥ 65% (should have raised) |

After 5+ decisions, the pattern with the highest rate (minimum 20%) becomes `primary_weakness`. This drives:
1. A banner shown at the top of every betting screen
2. A summary shown at game-over

### Progress file (`progress.json`)

Stored in the project root. Contains:
- `player_name`, `chips`, `starting_chips`
- `hands_played`, `hands_won`
- `session_history` — chip count after each hand
- `decision_stats` — raw counts for all 4 patterns
- `primary_weakness` — current identified weakness key
- `tutorial_seen` — whether to skip the tutorial on next launch

---

## Future: Ollama LLM integration

Phase 5 will add an Ollama call after each hand for a richer, conversational explanation. The planned integration point is `coaching.hand_review()` — it already returns structured decision data that can be formatted into a prompt.

Draft prompt template:
```
You are a poker coach. The player just finished a hand of Texas Hold'em.
Here are their decisions: {decisions}
Their identified weakness is: {weakness}
Give 2-3 sentences of personalized coaching. Be specific and encouraging.
```

Requires Ollama running locally (`ollama run llama3` or similar).
