# app/utils/prompting.py
from textwrap import dedent
from typing import List, Optional

TONE_STYLE = {
    "professional": "concise, objective, clinically precise; use plain English, not jargon; no emojis.",
    "caring": "warm, gentle, validating; use soft hedges like 'might/let's/if you're up for it'.",
    "balanced": "professional clarity with warm empathy; brief, human, and grounded.",
}

BANNED_DEFAULT = [
    "It's okay to feel",
    "I am here with you",
    "That sounds heavy",
    "Let's take a few deep breaths",
    "I'm sorry to hear that",
]

def _tone_style(tone: Optional[str]) -> str:
    t = (tone or "balanced").lower()
    return TONE_STYLE.get(t, TONE_STYLE["balanced"])

def build_therapist_prompt(
    *,
    context: str,
    question: str,
    history: str,
    tone: Optional[str],
    topics: str = "",
    preferred_tech: str = "",
    last_technique: str = "",
    next_step_index: int = 0,
    banned_phrases: Optional[List[str]] = None,
    fewshot: str = "",
) -> str:
    bans = list(dict.fromkeys((banned_phrases or []) + BANNED_DEFAULT))
    bans_str = "\n".join(f"- {b}" for b in bans)
    style = _tone_style(tone)

    core = dedent(f"""
    You are a licensed mental-health clinician. Sound like a real human, not a template.

    CONSTRAINTS
    - Output EXACTLY 3 lines:
      1) Empathy: ONE short sentence that paraphrases the user's words (no clichés).
      2) Step: ONE micro-step with explicit timing/reps (e.g., "inhale 4, hold 2, exhale 6 — 4 cycles").
      3) Q: ONE question only, prefixed with "Q:", preferably a 0–10 rating or permission to proceed.
    - ≤150 words. Complete sentences. No emojis. No bullet points.
    - If the user says "continue" or posts a 0–10 number, advance to the NEXT micro-step of the current technique.
    - Avoid these phrases and close paraphrases:
    {bans_str}

    TONE
    - Style: {style}
    - Natural conversational English with contractions. Vary openings each turn; never reuse the same first 5 words.

    TECHNIQUE CONTEXT
    - Topics: {topics or "n/a"}
    - Preferred technique: {preferred_tech or "n/a"}
    - Last technique: {last_technique or "n/a"}
    - Next step index: {next_step_index}

    KNOWLEDGE (use only if helpful; keep concise)
    {context}

    HISTORY (recent summary + last turns)
    {history}

    OUTPUT FORMAT (strict)
    Line 1: empathy (human, varied, paraphrase the user)
    Line 2: one actionable micro-step with timing/reps
    Line 3: Q: <one question>
    """).strip()

    if fewshot:
        core += "\n\n" + fewshot.strip()

    return core

def build_repair_prompt(*, raw: str, question: str, tone: str) -> str:
    style = _tone_style(tone)
    return dedent(f"""
    Rewrite the assistant reply to strictly follow the E/S/Q three-line format.

    RULES:
    - 3 lines only: Empathy, Step (timing/reps), Q: ...
    - ≤150 words total. Human, varied, no clichés.
    - Tone: {style}

    USER: {question}
    ASSISTANT_RAW:
    {raw}
    """).strip()

def build_minimal_esq_prompt(question: str, tone: str) -> str:
    style = _tone_style(tone)
    return dedent(f"""
    Produce a minimal E/S/Q reply (3 lines only) that still feels human and specific.

    USER: {question}
    TONE: {style}

    Output:
    - Line 1 Empathy (paraphrase the user)
    - Line 2 ONE micro-step with duration/reps
    - Line 3 Q: <one question>
    """).strip()
