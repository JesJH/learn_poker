# Learn to Play Poker

A Texas Hold'em poker game with built-in coaching — designed to teach you correct strategy through real-time tips, post-hand analysis, and adaptive feedback that tracks where you need the most improvement.

## Features

- **Play Texas Hold'em** against three AI opponents with distinct styles (Tight, Loose, Aggressive)
- **Coaching tips on demand** — request a tip before each decision to see hand quality, pot odds, position advice, and a clear recommendation
- **Post-action feedback** — graded instantly after every decision (✅ Good / 🟡 Ok / ❌ Mistake)
- **Post-hand review** — after each hand, key decision moments are flagged with explanations
- **Adaptive learning** — the game tracks your decision patterns over time and highlights your primary weakness (e.g. too passive, ignoring pot odds)
- **Progress saved** between sessions — chips, win rate, and decision history persist locally
- **Skippable tutorial** — hand rankings, poker terms, and how a hand works, always accessible in the sidebar

## How to run

```bash
pip install streamlit
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| UI | [Streamlit](https://streamlit.io) |
| Game engine | Custom (pure Python) |
| AI opponents | Rule-based (no external API) |
| Coaching | Rule-based strategy engine |
| Progress | Local JSON file |

## Project structure

```
learn_to_play_poker/
├── app.py                  # Streamlit UI and game state machine
├── progress.json           # Auto-created: saved player progress
└── poker/
    ├── card.py             # Card and Deck classes
    ├── player.py           # Player state (chips, hole cards, actions)
    ├── game.py             # Texas Hold'em game loop
    ├── ai.py               # Rule-based AI opponents
    ├── hand_evaluator.py   # Hand ranking (High Card → Royal Flush)
    ├── coaching.py         # Pre-action tips and post-hand analysis
    └── progress.py         # Progress persistence and adaptive tracking
```

## Roadmap

- [ ] Ollama integration for richer post-hand explanations via local LLM
- [ ] More AI opponent personalities
- [ ] Heads-up (1v1) mode
