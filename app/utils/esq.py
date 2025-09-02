# app/utils/esq.py
from __future__ import annotations
import re
from typing import Tuple

OUTPUT_CONTRACT = {"fields": ["E", "S", "Q"]}

_LABEL_LINE = re.compile(r"^\s*(E|S|Q)\s*:\s*(.*)$", re.I)
_ROLE_LINE  = re.compile(r"^\s*(system|assistant|user|human|message)\s*[:：]\s*", re.I)
CODE_FENCE  = re.compile(r"(?is)`{3}.*?`{3}")
SPLIT_SENT  = re.compile(r"(?<=[.!?])\s+")

# 新增：把規劃/旁白/Q&A腳手架砍掉
NOISY_PREFIXES = (
    "plan:", "encourage", "repaired reply:", "answer:", "question:",
    "natural chat", "note:", "instruction:", "assistant:", "human:", "user:", "message:",
)

START_FLAIR = re.compile(
    r"^(i(?:'| a)m (sorry|here)|thanks for|glad you|it'?s okay|we can|let'?s)\b",
    re.I,
)

def _strip_code_roles_noise(text: str) -> str:
    text = CODE_FENCE.sub("", text or "")
    lines = []
    for ln in text.splitlines():
        raw = _ROLE_LINE.sub("", ln).strip()
        low = raw.lower()
        if not raw:
            continue
        if any(low.startswith(p) for p in NOISY_PREFIXES):
            continue
        # 也排除「… Answer: …」「… Question: …」內嵌片段
        if " answer:" in low or " question:" in low:
            continue
        lines.append(raw)
    return "\n".join(lines)

def _extract_esq(raw: str) -> Tuple[str, str, str]:
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
    sentences = [x.strip() for x in SPLIT_SENT.split(raw.strip()) if x.strip()]
    if not sentences:
        return "", "", ""
    e = sentences[0]
    if len(sentences) > 1:
        s = sentences[1]
    if len(sentences) > 2:
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
    return " ".join(words[:max_words]).rstrip(",.;:! ")

def _drop_extra_questions(text: str) -> str:
    out, q_seen = [], False
    for ch in text:
        if ch == "?":
            out.append("?" if not q_seen else ".")
            q_seen = True
        else:
            out.append(ch)
    return "".join(out)

def _dedupe_sentences(text: str) -> str:
    seen, out = set(), []
    for s in [x.strip() for x in SPLIT_SENT.split(text) if x.strip()]:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key); out.append(s)
    return " ".join(out)

def _clean_ack(text: str, max_words: int = 10) -> str:
    t = _first_sentence(text)
    t = START_FLAIR.sub("", t).strip()
    if t.endswith("?"):
        t = t[:-1].rstrip()
    return _clip_words(t, max_words)

def _clean_suggestion(text: str, max_words: int = 14) -> str:
    t = _first_sentence(text)
    t = re.sub(r"\b(after that|now|then|first|second|next),?\b.*$", "", t, flags=re.I).strip()
    t = t.replace("?", ".").strip().rstrip(".")
    return _clip_words(t, max_words)

def _clean_question(text: str, max_words: int = 14) -> str:
    if not text:
        return "What feels manageable to try next?"
    parts = [p.strip() for p in SPLIT_SENT.split(text) if p.strip()]
    first_q = next((p for p in parts if p.endswith("?")), "")
    q = first_q or parts[0]
    q = _clip_words(q, max_words).rstrip(".! ")
    if not q.endswith("?"):
        q = "What feels manageable to try next?"
    return q

def format_esq(raw_text: str, output_contract: dict | None = None, word_limit: int = 28) -> str:
    txt = _strip_code_roles_noise(raw_text or "")
    e, s, q = _extract_esq(txt)
    ack = _clean_ack(e, 10)
    sug = _clean_suggestion(s, 14)
    ask = _clean_question(q, 14)
    pieces = [p for p in (ack, sug, ask) if p]
    out = ". ".join(pieces)
    out = _dedupe_sentences(out)
    out = _drop_extra_questions(out)
    end_q = out.endswith("?")
    out = _clip_words(out, word_limit)
    if end_q and not out.endswith("?"):
        out = out.rstrip(".! ") + "?"
    return out.strip()

def fallback_esq(question: str) -> str:
    base = "I’m here with you. Try a slow 4–2–6 breath once."
    ask = "What feels manageable to try next?"
    combo = f"{base} {ask}"
    return _clip_words(combo, 28)
