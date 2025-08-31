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
    bans_str = "; ".join(bans)
    style = _tone_style(tone)

    core = dedent(f"""
You are a licensed mental-health clinician. Your task is to write EXACTLY three lines (E, S, Q) that are specific, varied, and human.

CONSTRAINTS:
    - E: ONE short, specific empathy sentence that paraphrases the user. Rotate empathy openers and avoid repeating prior lines.
    - S: ONE concrete micro-step with explicit timing or repetitions (e.g., "inhale 4, hold 2, exhale 6 — 4 cycles").
    - Q: ONE brief question, preferably a 0–10 rating or asking permission to proceed.
    - The total response must be under 150 words. Use complete sentences. No emojis.
    - If the user says "continue" or gives a 0-10 rating, provide the NEXT step of the current technique.
    
AVOID CLICHÉS:
    - Do not use these phrases or close variations: {bans_str}

TONE:
    - Style: {style}
    - Use natural, conversational English with contractions.

TECHNIQUE CONTEXT
    - Topics: {topics or "n/a"}
    - Preferred technique: {preferred_tech or "n/a"}
    - Last technique: {last_technique or "n/a"}
    - Next step index: {next_step_index}

KNOWLEDGE (use only if helpful; keep concise)
    {context}

HISTORY (recent summary + last turns)
    {history}

OUTPUT FORMAT (strict):
    E: <empathy>
    S: <step>
    Q: <question>
    """ ).strip()

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
    """ ).strip()

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
    """ ).strip()

def build_reflection_prompt(question: str) -> str:
    """Builds a prompt for a reflective listening response."""
    return dedent(f"""
        You are a reflective listening assistant. Your task is to paraphrase the user's last message to show you are listening, not to solve the problem.
        The user's last message was: "{question}"

        Generate a three-line response following these rules:
        1. E-line: Start with empathy and paraphrase one or two key feelings or topics from the user's message.
        2. S-line: Do NOT suggest a new action or technique. Instead, briefly summarize the core content of what the user just said.
        3. Q-line: Ask a simple confirmation question like "Did I get that right?" or "Am I understanding you correctly?".
        
        Return the response in the strict E/S/Q format.
        E: <empathy and paraphrase>
        S: <summary of user's point>
        Q: Did I get that right?
    """ ).strip()