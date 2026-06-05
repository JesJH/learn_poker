"""
Concept card library for quant mode.

Each concept has:
  - title       : short name
  - formula     : LaTeX string (rendered with st.latex)
  - explanation : plain English, 2-3 sentences
  - trading     : how this maps to quantitative trading/finance
  - reference   : book or paper citation
  - challenge   : a question to ask the player at the next hand
  - tags        : which game situations trigger this concept

The teaching loop:
  When a concept is introduced, a challenge question is stored in session
  state. At the next relevant decision, the player is prompted to answer.
  Their answer is passed to the local LLM (Ollama) for evaluation and feedback.
"""
from __future__ import annotations

CONCEPTS: dict[str, dict] = {

    "expected_value": {
        "title": "Expected Value (EV)",
        "formula": r"EV = \sum_{i} p_i \cdot x_i",
        "explanation": (
            "Expected Value is the probability-weighted average of all possible outcomes. "
            "If you face a decision many times, EV tells you what you gain or lose on average. "
            "A positive-EV action is profitable long-term, even if it loses any given hand."
        ),
        "trading": (
            "In trading, EV is the expected P&L of a position: "
            "E[P&L] = p(win) × profit - p(loss) × loss. "
            "Every trade a quant firm places is evaluated on EV, not on whether it wins today."
        ),
        "reference": "Chen & Ankenman, The Mathematics of Poker (2006), Ch. 3",
        "challenge": (
            "In the next hand, before you act: calculate the EV of calling manually. "
            "You'll need: pot size, the amount to call, and your win probability (shown above). "
            "Formula: EV = pot × win% − to_call × (1 − win%). "
            "Type your calculation and what action it implies."
        ),
        "tags": ["preflop", "flop", "turn", "river"],
    },

    "pot_odds": {
        "title": "Pot Odds & Break-Even Equity",
        "formula": r"\text{Break-even equity} = \frac{\text{to\_call}}{\text{pot} + \text{to\_call}}",
        "explanation": (
            "Pot odds tell you the minimum equity needed to make a call profitable. "
            "If calling costs 20% of the pot, you need to win more than 20% of the time to profit. "
            "If your equity exceeds the pot odds, calling has positive EV."
        ),
        "trading": (
            "This is identical to the risk/reward ratio in trading. "
            "If a trade risks $1 to make $4, you need a >20% win rate to be profitable. "
            "Traders call this the 'minimum required hit rate.'"
        ),
        "reference": "Sklansky, Theory of Poker (1999), Ch. 4 — The Fundamental Theorem of Poker",
        "challenge": (
            "In the next hand where you face a bet: calculate the break-even equity yourself. "
            "Divide the call amount by (pot + call amount). "
            "Then compare to the Monte Carlo equity shown. Should you call? Why?"
        ),
        "tags": ["flop", "turn", "river"],
    },

    "kelly_criterion": {
        "title": "Kelly Criterion — Optimal Bet Sizing",
        "formula": r"f^* = \frac{b \cdot p - q}{b} = p - \frac{q}{b}",
        "explanation": (
            "Kelly answers: what fraction of your bankroll maximizes long-run growth? "
            "Betting more than Kelly is provably suboptimal — you'll grow slower or go broke. "
            "In practice, half-Kelly is preferred because p is never known exactly."
        ),
        "trading": (
            "Quantitative hedge funds use Kelly (or fractional Kelly) to size positions across "
            "trading signals. A signal with 60% win rate and 1:1 payoff → Kelly says bet 20% "
            "of capital. Overbetting destroys compounding; underbetting wastes edge."
        ),
        "reference": "Kelly, J.L. (1956). 'A New Interpretation of Information Rate.' Bell System Technical Journal.",
        "challenge": (
            "In the next hand where you must call: calculate the Kelly fraction manually. "
            "b = pot / to_call. f* = (b × win% − lose%) / b. "
            "What does Kelly recommend? How does it compare to your chip stack? "
            "Does calling for more than your Kelly fraction make sense?"
        ),
        "tags": ["flop", "turn", "river"],
    },

    "bayesian_updating": {
        "title": "Bayesian Updating — New Information Changes Everything",
        "formula": r"P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}",
        "explanation": (
            "Bayes' theorem describes how to update beliefs when new evidence arrives. "
            "Before the flop, all hands are roughly equally likely for an opponent. "
            "After they raise pre-flop, you should narrow their range to stronger hands. "
            "Each action is evidence that updates your model of what they hold."
        ),
        "trading": (
            "This is identical to how a trader updates a price forecast when new data arrives — "
            "earnings report, Fed announcement, order flow imbalance. "
            "Prior belief × likelihood of the new data → updated posterior belief. "
            "All Bayesian filters (Kalman filter in algo trading) use this exact structure."
        ),
        "reference": "Bayes, T. (1763). 'An Essay Towards Solving a Problem in the Doctrine of Chances.'",
        "challenge": (
            "Watch how your opponent bets this hand. After each action, ask yourself: "
            "does this raise or lower the probability they have a strong hand? "
            "At showdown, note if their revealed cards matched your updated belief. "
            "Type what you observed and how it changed your read."
        ),
        "tags": ["flop", "turn", "river"],
    },

    "sunk_cost": {
        "title": "Sunk Cost Fallacy",
        "formula": r"EV(\text{fold}) = 0 \quad \text{always (chips in pot are gone)}",
        "explanation": (
            "Chips already in the pot are gone regardless of what you do next. "
            "The decision to call or fold should depend only on future expected value, "
            "never on 'I've already put so much in.' That reasoning loses money long-term."
        ),
        "trading": (
            "This is one of the most costly biases in trading: holding a losing position "
            "because 'I'm already down' or averaging into a loser. "
            "Every position should be evaluated on its forward expected return, "
            "not its historical cost basis. 'Never marry a position.'"
        ),
        "reference": "Kahneman, D. (2011). Thinking, Fast and Slow. Ch. 32 — Keeping Score.",
        "challenge": (
            "In the next hand where you've invested chips: before calling, ask yourself — "
            "'If I hadn't put those chips in yet, would I still call?' "
            "If no, you're about to make the sunk cost mistake. Type your honest answer."
        ),
        "tags": ["flop", "turn", "river"],
    },

    "variance_risk": {
        "title": "Variance & Risk of Ruin",
        "formula": r"\text{Risk of Ruin} \approx \left(\frac{q}{p}\right)^{B/u}",
        "explanation": (
            "Even with a positive edge, high variance can wipe out your bankroll before "
            "your edge manifests. A coin that pays 3:1 is clearly +EV, but if you bet "
            "everything every time, eventual ruin is guaranteed. "
            "Bankroll management (Kelly) exists to avoid this."
        ),
        "trading": (
            "Risk of ruin is why quantitative strategies use strict position limits. "
            "A strategy with Sharpe ratio 1.0 can still have 20-40% drawdowns. "
            "Firms size positions so a bad streak doesn't end the strategy before "
            "the edge has time to compound."
        ),
        "reference": "Poundstone, W. (2005). Fortune's Formula. Hill and Wang. Ch. 3.",
        "challenge": (
            "Consider your current chip stack versus your starting stack. "
            "What percentage have you lost or gained? At what chip count would you "
            "need to stop and reload? How does that relate to Kelly bet sizing — "
            "should you be betting more or less aggressively given your current stack?"
        ),
        "tags": ["preflop"],
    },

    "decision_node": {
        "title": "Decision Node Analysis (Game Tree)",
        "formula": r"V(\text{node}) = \max_a \sum_o P(o \mid a) \cdot V(\text{child}_{a,o})",
        "explanation": (
            "A game tree maps every possible sequence of actions and outcomes. "
            "At each decision node, optimal play chooses the action with highest expected value "
            "across all possible futures. This backward-induction is called dynamic programming."
        ),
        "trading": (
            "This is identical to pricing American options (early exercise is a decision node) "
            "or dynamic portfolio rebalancing — both solved with backward induction on a "
            "tree of possible market states. Binomial option pricing is literally this."
        ),
        "reference": "Von Neumann, J. & Morgenstern, O. (1944). Theory of Games and Economic Behavior.",
        "challenge": (
            "After this hand, draw (mentally or on paper) the decision tree for one street. "
            "What were your possible actions? What could have happened after each? "
            "Which branch had the highest EV in hindsight? "
            "Type your tree description and conclusion."
        ),
        "tags": ["river"],
    },
}


# Which concept to introduce based on game situation
CONCEPT_SEQUENCE = [
    "expected_value",       # hand 1 — the foundation of everything
    "pot_odds",             # hand 2 — apply EV to calling decisions
    "kelly_criterion",      # hand 3 — optimal sizing given your edge
    "sunk_cost",            # hand 4 — common mistake to eliminate
    "bayesian_updating",    # hand 5 — reading opponents / new info
    "variance_risk",        # hand 6 — why bankroll management matters
    "decision_node",        # hand 7 — game tree / dynamic programming
]


def concept_for_hand(hand_number: int) -> dict | None:
    """Return the concept to introduce this hand (cycles through the sequence)."""
    idx = (hand_number - 1) % len(CONCEPT_SEQUENCE)
    key = CONCEPT_SEQUENCE[idx]
    return {"key": key, **CONCEPTS[key]}


def get_concept(key: str) -> dict | None:
    return CONCEPTS.get(key)
