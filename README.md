# Learn Poker

A Texas Hold'em poker learning game with two modes: 

**Standard** for learning poker strategy,  

**Quant Trading** for understanding the math behind every decision. Built with Python and Streamlit — playable in your browser, no install needed.

🃏 **[Click Link to Play →](https://learn-poker.streamlit.app)**

---

## Two ways to play

### 🃏 Standard Mode — Learn poker strategy
Play Texas Hold'em against three AI opponents and get coached on every decision. Designed for beginners who want to understand *why* a move is good or bad.

- Hover over the coaching tip before you act to see hand quality, pot odds, and position advice
- Get instant feedback after every action (✅ Good / 🟡 Ok / ❌ Mistake)
- See a full hand review at the end — what went well, the risk you took, and the EV context for each decision
- If you folded, see exactly what hand you would have made so you can study it
- Ask the AI coach any question mid-hand in plain English
- The game tracks your weaknesses over time and surfaces them as reminders

### 🔬 Quant Trading Mode — Learn the math behind the decisions
The same Texas Hold'em game, but every hand is a lesson in quantitative finance. Each decision comes with live analysis connecting poker to trading theory.

| Poker concept | Quant / trading equivalent |
|---|---|
| Hand equity | Position sizing based on win probability |
| Pot odds | Break-even analysis on a trade |
| Expected value | EV of entering vs. passing a trade |
| Kelly Criterion | Optimal bet sizing to maximise long-run growth |
| Bayesian updating | Revising your read as new cards appear |

Each hand introduces one concept with a formula, a plain-English explanation, and a trading analogy. The next hand challenges you to apply it.

#### Kuhn Poker Lab (inside Quant mode)
A stripped-down 3-card poker game where the Nash Equilibrium is mathematically solvable. Use it to understand Game Theory Optimal (GTO) play — the strategy a rational market maker uses — and see the cost of deviating from it.

---

## Features

- **Visual poker table** — opponents sit across a green felt, chips and blinds (D / SB / BB) shown in each seat
- **Coaching tip on hover** — see hand quality, pot odds, position advice, and a recommendation without clicking
- **Weak hand guide** — when you hold a marginal hand, a decision checklist auto-expands to walk you through what to consider
- **Folded hand reveal** — at the end of every hand, see the cards you folded and what hand you would have made
- **Post-hand review** — per-street breakdown: what went well, the risk, and EV context for each decision
- **Adaptive learning** — tracks fold/call/raise patterns over time and flags your primary leak
- **Progress saved** between sessions — chips, win rate, and history persist locally
- **AI coach** — ask anything mid-hand (requires Ollama running locally)

---

## Run locally

```bash
pip install streamlit
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Optional — AI coach via Ollama:**
```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3
ollama serve
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| UI | [Streamlit](https://streamlit.io) |
| Game engine | Custom (pure Python) |
| AI opponents | Rule-based (Tight / Loose / Aggressive) |
| Coaching | Rule-based strategy engine |
| Quant analysis | Monte Carlo equity, EV calculator, Kelly Criterion |
| AI coach | Ollama (local LLM, optional) |
| Progress | Local JSON file |

## Project structure

```
learn_to_play_poker/
├── app.py                    # Streamlit UI and game state machine
├── progress.json             # Auto-created: saved player progress
└── poker/
    ├── card.py               # Card and Deck classes
    ├── player.py             # Player state (chips, hole cards, actions)
    ├── game.py               # Texas Hold'em game loop
    ├── ai.py                 # Rule-based AI opponents
    ├── hand_evaluator.py     # Hand ranking (High Card → Royal Flush)
    ├── coaching.py           # Pre-action tips and post-hand review
    ├── progress.py           # Progress persistence and adaptive tracking
    ├── monte_carlo.py        # Equity simulation
    ├── ev_calculator.py      # Expected value at each decision node
    ├── kelly.py              # Kelly Criterion bet sizing
    ├── quant_concepts.py     # Per-hand concept cards (formula + explanation)
    ├── llm_coach.py          # Ollama AI coach integration
    └── kuhn_poker.py         # Kuhn Poker Lab (GTO solver)
```
