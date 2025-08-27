# app/utils/esq.py
# E/S/Q formatting utilities + greeting-aware, emotion-aware smart fallback.

import re
from typing import List, Optional

# ----------------------- Output contract (reference text) -----------------------
OUTPUT_CONTRACT = (
    "[OUTPUT CONTRACT]\n"
    "Return EXACTLY three lines, then <END>.\n"
    "E: <one short empathy sentence (≤20 words, varied from last turn)>\n"
    "S: <ONE micro-step with explicit duration/repetitions>\n"
    "Q: <ONE brief question; prefer a 0–10 rating>\n"
    "<END>"
)

# ------------------------------ Core formatter --------------------------------
_E_PAT = re.compile(r'(?mi)^[^\S\r\n]*E\s*[:：\-–]\s*(.+)$')
_S_PAT = re.compile(r'(?mi)^[^\S\r\n]*S\s*[:：\-–]\s*(.+)$')
_Q_PAT = re.compile(r'(?mi)^[^\S\r\n]*Q\s*[:：\-–]\s*(.+)$')

def _clean(s: str) -> str:
    return " ".join((s or "").strip().split())

def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def _truncate_words(s: str, max_words: int) -> str:
    words = re.findall(r"\b\w+\b", s or "")
    if len(words) <= max_words:
        return (s or "").strip()
    return " ".join(words[:max_words]).rstrip(",.;:") + "."

def format_esq(raw: str, word_limit: int = 120) -> str:
    """Parse raw model text into strict three-line E/S/Q; return '' if not found."""
    if not raw:
        return ""
    E = _E_PAT.search(raw)
    S = _S_PAT.search(raw)
    Q = _Q_PAT.search(raw)
    if not (E and S and Q):
        return ""

    e = _clean(E.group(1))
    s = _clean(S.group(1))
    q = _clean(Q.group(1))
    if not q.endswith("?"):
        q = q.rstrip(". ") + "?"

    # Enforce total word limit (trim S first, then E if still long)
    total = _word_count(" ".join([e, s, q]))
    if total > word_limit:
        overflow = total - word_limit
        s_words = _word_count(s)
        cut = min(overflow, max(0, s_words - 3))
        if cut > 0:
            s = _truncate_words(s, max(1, s_words - cut))
        total = _word_count(" ".join([e, s, q]))
        if total > word_limit:
            e = _truncate_words(e, max(3, _word_count(e) - (total - word_limit)))

    return f"E: {e}\nS: {s}\nQ: {q}"

# ------------------------------ Simple fallback --------------------------------
def fallback_esq(question: str) -> str:
    step = "Name 5 things you can see — 60 seconds."
    return (
        "E: That sounds heavy, and I am here with you.\n"
        f"S: {step}\n"
        "Q: After that, what is your tension level from 0–10?"
    )

# ------------------------------ Smart fallback --------------------------------
# Step maps for techniques used when we must produce a safe answer without model help.
STEP_MAP = {
    "grounding": [
        "Name 5 things you can see — 60 seconds.",
        "Touch 4 different textures — 60 seconds.",
        "Listen for 3 distinct sounds — 30 seconds.",
        "Notice 2 smells — 30 seconds.",
        "Name 1 thing you can taste — 10 seconds."
    ],
    "breathing": [
        "Inhale 4, hold 2, exhale 6 — 4 cycles.",
        "Repeat the same pattern — 4 cycles.",
        "Place a hand on your belly and feel the rise/fall for 4 breaths."
    ],
    "pmr": [
        "Tense hands 5s, release 10s — 2 times.",
        "Tense shoulders 5s, release 10s — 2 times.",
        "Tense jaw 5s, release 10s — 2 times."
    ],
    "stimulus_control": [
        "If awake >20 min, leave bed; do a quiet task 10–15 min.",
        "Return to bed when sleepy; repeat once if needed."
    ],
}

TOPIC_TO_TECHNIQUE = {
    "sleep": "stimulus_control",
    "panic": "breathing",
    "trauma": "grounding",
    "grief": "grounding",
    "anxiety": "grounding",
    "depress": "pmr",
}

_GREETING_ONLY = re.compile(r"^\s*(hi|hello|hey)\s*[!.]*\s*$", re.I)

EMPATHY_BY_MOOD = {
    "sad": [
        "I’m sorry this feels heavy, and I’m here with you.",
        "This sounds tough, and I’m with you."
    ],
    "anxious": [
        "It makes sense you feel on edge; I’m here with you.",
        "That sounds tense; I’m with you."
    ],
    "panic": [
        "That surge can be scary; I’m here with you.",
        "That sounds intense; I’m with you."
    ],
    "neutral": [
        "I’m glad you reached out; I’m here with you.",
        "I’m here with you; we can take this one step at a time."
    ],
}

def is_greeting(text: str) -> bool:
    return bool(_GREETING_ONLY.match((text or "").strip()))

def choose_empathy(emotion: Optional[str], greeting: bool) -> str:
    if greeting:
        return "I’m glad you reached out; I’m here with you."
    mood = (emotion or "").lower()
    if "panic" in mood:
        bank = EMPATHY_BY_MOOD["panic"]
    elif any(k in mood for k in ["anxiety", "anxious", "worry"]):
        bank = EMPATHY_BY_MOOD["anxious"]
    elif any(k in mood for k in ["sad", "low", "down", "depress"]):
        bank = EMPATHY_BY_MOOD["sad"]
    else:
        bank = EMPATHY_BY_MOOD["neutral"]
    return bank[0]

def _rating_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"\b(?:10|[0-9](?:\.\d)?)\b", text.strip())
    try:
        return float(m.group(0)) if m else None
    except Exception:
        return None

def _pick_technique(topics: List[str], last_technique: Optional[str], has_numeric: bool) -> str:
    if has_numeric and last_technique:
        return last_technique
    if topics:
        t = (topics[0] or "").lower()
        if t in TOPIC_TO_TECHNIQUE:
            return TOPIC_TO_TECHNIQUE[t]
    return last_technique or "grounding"

def _step_for(technique: str, index: int) -> str:
    steps = STEP_MAP.get(technique, STEP_MAP["grounding"])
    if index < 1:
        index = 1
    if index > len(steps):
        index = len(steps)
    return steps[index - 1]

def smart_fallback_esq(
    question: str,
    *,
    topics: Optional[List[str]] = None,
    last_technique: Optional[str] = None,
    step_index: int = 0,
    emotion: Optional[str] = None,
    is_greeting: bool = False,
) -> str:
    """
    Context-aware fallback:
    - If greeting-only on first turn, use gentle onboarding.
    - Else, choose technique from topics or continue last technique when a 0–10 rating is given.
    - Advance step when rating is present and technique matches.
    - Always return strict E/S/Q (three lines).
    """
    if is_greeting or is_greeting_text(question := (question or "")):
        empathy = choose_empathy(emotion, greeting=True)
        step = "Take one slow breath — inhale 4, exhale 6 — 1 cycle."
        qline = "Q: What would you like to focus on today?"
        return f"E: {empathy}\nS: {step}\n{qline}"

    rating = _rating_from_text(question)
    has_numeric = rating is not None
    technique = _pick_technique(topics or [], last_technique, has_numeric)
    next_index = (step_index + 1) if (has_numeric and last_technique == technique and step_index >= 1) else 1
    step = _step_for(technique, next_index)
    empathy = choose_empathy(emotion, greeting=False)
    qline = (
        "Q: Would you like to proceed to the next step?"
        if has_numeric else
        "Q: After that, what is your tension level from 0–10?"
    )
    return f"E: {empathy}\nS: {step}\n{qline}"

# Helper to avoid shadowing param name in smart_fallback_esq
def is_greeting_text(text: str) -> bool:
    return is_greeting(text)

import re
def humanize_esq(text: str) -> str:
    t = text
    t = re.sub(r"\bIt(?:'| i)s okay to feel\b", "It makes sense to feel", t, flags=re.I)
    t = re.sub(r"\bNotice your breath\b", "Gently notice your breath", t, flags=re.I)
    t = re.sub(r"\bBreathe deeply for\b", "Let’s take a slow breath for", t, flags=re.I)
    parts = t.strip().splitlines()
    if parts and parts[-1].startswith("Q:"):
        import re as _re
        parts[-1] = _re.sub(r"[.!]+$", "", parts[-1]).rstrip() + "?"
        t = "\n".join(parts)
    return t