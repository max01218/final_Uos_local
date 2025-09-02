# app/utils/esq.py
from __future__ import annotations
import re
from typing import Tuple

# Minimal contract placeholder to keep imports working.
OUTPUT_CONTRACT = {"fields": ["E", "S", "Q"]}

_LABEL_LINE = re.compile(r"^\s*(E|S|Q)\s*:\s*(.*)$", re.I)
_ROLE_LINE = re.compile(r"^\s*(system|assistant|user|human|message)\s*[:：]\s*", re.I)
CODE_FENCE = re.compile(r"(?is)`{3}.*?`{3}")

SPLIT_SENT = re.compile(r"(?<=[.!?])\s+")

# Common filler to strip from starts.
START_FLAIR = re.compile(
    r"^(i(?:'| a)m (sorry|here)|thanks for|glad you|it'?s okay|we can|let'?s)\b",
    re.I,
)

def _strip_code_and_roles(text: str) -> str:
    text = CODE_FENCE.sub("", text or "")
    lines = []
    for ln in text.splitlines():
        ln = _ROLE_LINE.sub("", ln).strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)

def _extract_esq(raw: str) -> Tuple[str, str, str]:
    """Extract E/S/Q lines. If missing, derive heuristically."""
    e = s = q = ""
    for ln in (raw or "").splitlines():
        m = _LABEL_LINE.match(ln.strip())
        if not m:
            continue
        lab, rest = m.group(1).upper(), m.group(2).strip()
        if lab == "E" and not e:
            e = rest
        elif lab == "S" and not s:
            s = rest
        elif lab == "Q" and not q:
            q = rest

    if e or s or q:
        return e, s, q

    # Heuristic fallback: map first 3 sentences to E/S/Q-ish.
    sentences = [x.strip() for x in SPLIT_SENT.split(raw.strip()) if x.strip()]
    if not sentences:
        return "", "", ""
    e = sentences[0]
    if len(sentences) > 1:
        s = sentences[1]
    if len(sentences) > 2:
        # pick first sentence that ends with '?', else the third one
        cand_q = next((t for t in sentences[1:] if t.endswith("?")), sentences[2])
        q = cand_q
    return e, s, q

def _first_sentence(text: str) -> str:
    if not text:
        return ""
    return SPLIT_SENT.split(text.strip())[0].strip()

def _clip_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",.;:! ")  # keep end clean

def _drop_extra_questions(text: str) -> str:
    # Keep only the first question. Convert extra '?' to '.'.
    out = []
    q_seen = False
    for ch in text:
        if ch == "?":
            if q_seen:
                out.append(".")
            else:
                q_seen = True
                out.append("?")
        else:
            out.append(ch)
    return "".join(out)

def _dedupe_sentences(text: str) -> str:
    seen = set()
    out = []
    for s in [x.strip() for x in SPLIT_SENT.split(text) if x.strip()]:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return " ".join(out)

def _clean_ack(text: str, max_words: int = 12) -> str:
    t = _first_sentence(text)
    t = START_FLAIR.sub("", t).strip()
    # remove trailing question if any in E
    if t.endswith("?"):
        t = t[:-1].rstrip()
    return _clip_words(t, max_words)

def _clean_suggestion(text: str, max_words: int = 16) -> str:
    # Take only the first actionable clause/sentence.
    t = _first_sentence(text)
    # trim common sequencers to reduce length
    t = re.sub(r"\b(after that|now|then|and then|first|second|next),?\b.*$", "", t, flags=re.I).strip()
    # Avoid ending with '?' in S
    t = t.replace("?", ".").strip().rstrip(".")
    return _clip_words(t, max_words)

def _clean_question(text: str, max_words: int = 16) -> str:
    # Keep the first question-only sentence; if missing, convert sentence to a short question.
    if not text:
        return "What feels manageable to try next?"
    # pick first question mark span
    parts = [p.strip() for p in SPLIT_SENT.split(text) if p.strip()]
    first_q = next((p for p in parts if p.endswith("?")), "")
    q = first_q or parts[0]
    q = _clip_words(q, max_words).rstrip(".! ")
    if not q.endswith("?"):
        # standardize to a single, short question
        q = "What feels manageable to try next?"
    return q

def format_esq(raw_text: str, output_contract: dict | None = None, word_limit: int = 28) -> str:
    """
    Convert E/S/Q into ONE short chat message:
    - 1 short acknowledgment (E)
    - 1 concrete, low-burden suggestion (S)
    - EXACTLY ONE short question (Q, ends with '?')
    Enforces single-question policy and overall word limit.
    """
    txt = _strip_code_and_roles(raw_text or "")
    e, s, q = _extract_esq(txt)

    ack = _clean_ack(e, 12)
    sug = _clean_suggestion(s, 16)
    ask = _clean_question(q, 16)

    # Build surface: avoid empty pieces, ensure punctuation.
    pieces = []
    if ack:
        pieces.append(ack)
    if sug:
        pieces.append(sug)
    if ask:
        pieces.append(ask)

    out = ". ".join(pieces)
    out = _dedupe_sentences(out)
    out = _drop_extra_questions(out)

    # Enforce global word limit; keep the question mark at end.
    end_q = out.endswith("?")
    out = _clip_words(out, word_limit)
    if end_q and not out.endswith("?"):
        # restore '?', but keep single-question policy
        out = out.rstrip(".! ") + "?"
    return out.strip()

def fallback_esq(question: str) -> str:
    # Safe minimal fallback; one suggestion + one question.
    base = "I’m here with you. Try a slow 4–2–6 breath once."
    ask = "What feels manageable to try next?"
    combo = f"{base} {ask}"
    return _clip_words(combo, 28)
