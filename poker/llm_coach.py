"""
Ollama LLM coach integration.

Uses a system prompt to establish the coach persona and context, then passes
the specific game state + player answer as the user message so the model can
verify calculations against real numbers from the hand.

Ollama API: POST http://localhost:11434/api/generate
  { "model": ..., "system": ..., "prompt": ..., "stream": false }
"""
from __future__ import annotations
import json
import urllib.request

MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 45  # seconds

SYSTEM_PROMPT_GRADE = """You are a quant trading coach. Reply in exactly 3 lines:
Line 1: RIGHT or WRONG — one sentence on whether the math is correct.
Line 2: The correct calculation using the real numbers provided, nothing else.
Line 3: One sentence connecting this to trading. No greetings, no padding."""

SYSTEM_PROMPT_EXPLAIN = """You are a quant trading and poker coach. Answer concisely in 3-5 sentences.
Give the formula, what it means in plain English, and one concrete trading example.
No padding, no bullet lists unless the question requires it."""


def evaluate_challenge(
    concept: dict,
    player_answer: str,
    game_context: dict,
) -> str:
    """
    Send player's challenge answer to Ollama for grading.

    Args:
        concept       : the concept dict from quant_concepts.py
        player_answer : what the player typed
        game_context  : dict with pot, to_call, equity, ev_call, ev_raise, kelly

    Returns:
        Feedback string from the LLM, or an error message if Ollama is down.
    """
    ctx = game_context

    # Build a user message that includes the real numbers so the LLM can verify math
    user_prompt = f"""Concept being evaluated: {concept['title']}
Formula: {concept['formula']}

Real game state for this hand:
- Pot size: ${ctx.get('pot', 'unknown')}
- Amount to call: ${ctx.get('to_call', 'unknown')}
- Monte Carlo equity (win probability): {round(ctx.get('equity', 0) * 100, 1)}%
- Correct EV(call) = pot × equity − to_call × (1 − equity) = ${ctx.get('ev_call', 'unknown')}
- Correct EV(raise) ≈ ${ctx.get('ev_raise', 'unknown')} (assumes 30% fold equity)
- Kelly fraction = {round(ctx.get('kelly_fraction', 0) * 100, 1) if ctx.get('kelly_fraction') else 'n/a'}%

Challenge question asked:
{concept['challenge']}

Student's answer:
{player_answer if player_answer.strip() else '[No answer provided — student left it blank]'}

Evaluate the student's answer. Check their arithmetic against the correct numbers above.
If they skipped the math, push them to engage with it next time. Connect to trading."""

    try:
        payload = json.dumps({
            "model": MODEL,
            "system": SYSTEM_PROMPT_GRADE,
            "prompt": user_prompt,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            result = json.loads(resp.read())
            return result.get("response", "No response received from Ollama.")

    except urllib.error.URLError:
        return (
            "**Ollama is not running.** Start it by opening the Ollama app or running "
            "`ollama serve` in a terminal, then try again. "
            f"Model needed: `{MODEL}` — run `ollama pull {MODEL}` if not yet downloaded."
        )
    except Exception as e:
        return f"Error communicating with Ollama: {e}"


def post_hand_analysis(
    hand_decisions: list[dict],
    player_name: str,
    community_cards: list,
    weakness: dict | None,
    game_context: dict,
) -> str:
    """
    Generate a post-hand narrative coaching note from the full hand history.
    Called after showdown when the player clicks 'Ask AI coach'.
    """
    decisions_text = "\n".join(
        f"- {d['street']}: {d['action']} (hole: {d['hole_cards']}, community: {d['community']})"
        for d in hand_decisions
        if d["player"] == player_name
    )

    weakness_text = (
        f"The player's identified long-term weakness is: {weakness['label']}. {weakness['tip']}"
        if weakness else "No weakness pattern identified yet (fewer than 5 decisions tracked)."
    )

    user_prompt = f"""Hand: board={community_cards}, equity={round(game_context.get('equity', 0)*100,1)}%, pot=${game_context.get('pot','?')}
Decisions: {decisions_text if decisions_text else 'folded pre-flop'}
Weakness: {weakness_text}

Reply in exactly 2 lines:
Line 1: The single decision that most hurt or helped — name the street and action, say why in one sentence.
Line 2: How it links to their weakness pattern. One sentence only."""

    try:
        payload = json.dumps({
            "model": MODEL,
            "system": SYSTEM_PROMPT_GRADE,
            "prompt": user_prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            result = json.loads(resp.read())
            return result.get("response", "No response received.")
    except urllib.error.URLError:
        return "Ollama is not running — start it to enable post-hand AI analysis."
    except Exception as e:
        return f"Error: {e}"


def explain_concept(question: str) -> str:
    """Answer an open-ended concept question with no format constraints."""
    try:
        payload = json.dumps({
            "model": MODEL,
            "system": SYSTEM_PROMPT_EXPLAIN,
            "prompt": question,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read()).get("response", "No response.")
    except urllib.error.URLError:
        return "Ollama is not running — start it to enable AI explanations."
    except Exception as e:
        return f"Error: {e}"


def is_running() -> bool:
    """Quick health check — returns True if Ollama is reachable."""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False
