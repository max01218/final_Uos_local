# app/utils/esq.py
import re

OUTPUT_CONTRACT = "ESQ_THREE_LINES"

def _lines3(s: str):
    lines = (s or "").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]

def format_esq(raw_text: str, output_contract: str, word_limit: int = 45) -> str:
    if output_contract != "ESQ_THREE_LINES":
        return (raw_text or "").strip()

    text = (raw_text or "").strip()
    if not text:
        return ""

    text = re.sub(r"(?is)`{3}.*?`{3}", "", text)
    text = re.sub(r"(?i)\b(system|assistant|human|message)\b\s*[:：]\s*", "", text)

    def _pick(label: str) -> str:
        m = re.search(rf"(^|\n)\s*{label}\s*:\s*(.+)", text, re.I)
        if not m:
            return ""
        seg = m.group(2).strip()
        seg = re.split(r"(^|\n)\s*[ESQ]\s*:\s*", seg, maxsplit=1, flags=re.I)[0].strip()
        return seg

    e = _pick("E")
    s = _pick("S")
    q = _pick("Q")

    if not e or not s or not q:
        raise ValueError("Missing E/S/Q labels")

    def _limit(t: str) -> str:
        words = t.split()
        if len(words) > word_limit:
            return " ".join(words[:word_limit]).rstrip(".,;:! ")
        return t

    e, s, q = _limit(e), _limit(s), _limit(q)
    return f"E: {e}\nS: {s}\nQ: {q}"

def fallback_esq(question: str) -> str:
    e = "I’m here with you; we can take this one step at a time."
    s = "Try a short grounding: name 5 things you can see in the room."
    q = "After that, what feels most manageable to try next?"
    return f"E: {e}\nS: {s}\nQ: {q}"

_ESQ_LINE = re.compile(r"^\s*([ESQ])\s*:\s*(.+)$", re.I)
_ROLE_PREFIX = re.compile(r"^\s*(system|assistant|user|human|message)\s*[:：]\s*", re.I)

def parse_esq_lines(text: str) -> dict:
    """
    Parse lines labeled E:/S:/Q: into a dict like {'E': '...', 'S': '...', 'Q': '...'}.
    Ignores non-labeled lines. Returns empty dict if nothing parsed.
    """
    data = {}
    if not text:
        return data
    for raw in text.splitlines():
        line = (_ROLE_PREFIX.sub("", raw or "")).strip()
        m = _ESQ_LINE.match(line)
        if m:
            data[m.group(1).upper()] = m.group(2).strip()
    return data

def esq_to_natural(text: str, max_words: int = 120) -> str:
    """
    Convert E:/S:/Q: labeled text into ONE natural chat message:
    - keep E as an empathetic first sentence
    - keep S as one concrete, low-effort suggestion sentence
    - keep Q as exactly one short question at the end
    - no labels, no role prefixes, trimmed to ~max_words
    Fallback: if parsing fails, return the original text trimmed.
    """
    data = parse_esq_lines(text or "")
    e = (data.get("E") or "").strip()
    s = (data.get("S") or "").strip()
    q = (data.get("Q") or "").strip()

    parts = []
    if e:
        e = e.rstrip(".!?")
        parts.append(e + ".")
    if s:
        s = s.rstrip(".!?")
        parts.append(s + ".")
    if q:
        q = q.rstrip(".! ")
        if not q.endswith("?"):
            q = q + "?"
        parts.append(q)

    out = " ".join([p for p in parts if p]).strip()
    if not out:
        out = (text or "").strip()

    # clip words softly
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]).rstrip(".!?") + "."

    return out