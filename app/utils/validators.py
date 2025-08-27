
from __future__ import annotations
import re
from typing import Dict, Optional

_E_LABEL = re.compile(r"^\s*E\s*[:：-]\s*", re.I)
_S_LABEL = re.compile(r"^\s*S\s*[:：-]\s*", re.I)
_Q_LABEL = re.compile(r"^\s*Q\s*[:：-]\s*", re.I)
_GREET = re.compile(r"\b(hi|hello|hey|dear)\b", re.I)
_NAMEY = re.compile(r"\b(user|client|patient|sir|madam)\b", re.I)

# Duration/repetition cues (expand as needed)
_DURATION = re.compile(
    r"\b(\d{1,3}\s*(seconds?|secs?|s|minutes?|mins?|breaths?|cycles?|times?))\b",
    re.I,
)

def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def _normalize_lines(text: str) -> list[str]:
    lines = [l.rstrip() for l in (text or "").splitlines()]
    return [l for l in lines if l.strip()]

def _extract_lines(text: str) -> Dict[str, str]:
    """Return dict with raw E/S/Q lines (without leading label)."""
    lines = _normalize_lines(text)
    out = {"E": "", "S": "", "Q": ""}
    for l in lines:
        if _E_LABEL.match(l) and not out["E"]:
            out["E"] = _E_LABEL.sub("", l).strip()
        elif _S_LABEL.match(l) and not out["S"]:
            out["S"] = _S_LABEL.sub("", l).strip()
        elif _Q_LABEL.match(l) and not out["Q"]:
            out["Q"] = _Q_LABEL.sub("", l).strip()
    return out

def _variation_ok(curr: str, prev: Optional[str], thr: float = 0.85) -> bool:
    """Shallow anti-duplication: if prev exists, avoid near-identical strings."""
    if not curr or not prev:
        return True
    a = re.sub(r"\s+", " ", curr.strip().lower())
    b = re.sub(r"\s+", " ", prev.strip().lower())
    if a == b:
        return False
    # character-level overlap
    inter = len(set(a) & set(b))
    denom = max(len(set(a) | set(b)), 1)
    return (inter / denom) < thr

def validate_esq(text: str, last_assistant: Optional[str] = None) -> Dict[str, object]:
    """
    Validate an E/S/Q response. Returns detailed booleans plus a score.
    - exactly 3 non-empty lines
    - labels E:/S:/Q: present (tolerant to ： and -)
    - empathy line <= 20 words
    - micro-step contains explicit duration or repetitions
    - final line starts with Q: and contains a question mark
    - no greetings/names in empathy line
    - total words <= 120
    - variation vs previous assistant empathy/question (if provided)
    """
    lines = _normalize_lines(text)
    exactly_three = len(lines) == 3

    has_e = any(_E_LABEL.match(l) for l in lines)
    has_s = any(_S_LABEL.match(l) for l in lines)
    has_q = any(_Q_LABEL.match(l) for l in lines)

    parts = _extract_lines(text)
    E, S, Q = parts["E"], parts["S"], parts["Q"]

    empathy_words_ok = _word_count(E) <= 20 and _word_count(E) >= 3
    microstep_has_duration = bool(_Duration_or_reps(S=S))
    q_starts_ok = bool(Q) and lines[-1].startswith("Q:")
    q_has_mark = "?" in (Q or "")
    no_greeting = not (_GREET.search(E or "") or _NAMEY.search(E or ""))
    total_words_ok = _word_count(" ".join([E, S, Q])) <= 120

    # previous-turn anti-dup check
    prev_lines = _normalize_lines(last_assistant or "")
    prev_E = prev_lines[0] if prev_lines and prev_lines[0].startswith("E:") else ""
    prev_Q = prev_lines[-1] if prev_lines and prev_lines[-1].startswith("Q:") else ""
    var_e_ok = _variation_ok("E: " + (E or ""), prev_E)
    var_q_ok = _variation_ok("Q: " + (Q or ""), prev_Q)

    checks = {
        "exactly_three_lines": exactly_three,
        "has_E": has_e,
        "has_S": has_s,
        "has_Q": has_q,
        "empathy_word_limit_ok": empathy_words_ok,
        "microstep_has_duration_or_reps": microstep_has_duration,
        "question_line_starts_with_Q": q_starts_ok,
        "question_contains_mark": q_has_mark,
        "no_greeting_or_name": no_greeting,
        "total_word_limit_ok": total_words_ok,
        "variation_empathy_ok": var_e_ok,
        "variation_question_ok": var_q_ok,
    }
    score = sum(1 for v in checks.values() if v) / max(len(checks), 1)
    checks["score"] = round(score, 3)
    checks["all_ok"] = all(checks.values())
    return checks

def _Duration_or_reps(S: str) -> bool:
    """Return True if micro-step mentions explicit time or repetitions."""
    if not S:
        return False
    if _DURATION.search(S):
        return True
    # simple patterns: x cycles, times, breaths (without explicit number-unit spacing)
    return bool(re.search(r"\b(\d+\s*-\s*\d+\s*cycles|\d+\s*cycles|\d+\s*breaths?|\d+\s*times?)\b", S, re.I))
