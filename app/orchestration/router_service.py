import re
import logging
from dataclasses import dataclass

from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)

# ---------- 強訊號啟發式（更窄更準） ----------
DEFN_CUES = re.compile(
    r"\b(what\s+is|what's|define|definition|explain|explanation|"
    r"difference\s+between|symptoms\s+of|signs\s+of|how\s+does\s+.*\bwork)\b",
    re.I,
)

# 盡量涵蓋常見 MH 描述，但避免把「解釋題」誤判（由 DEFN_CUES 先攔）
MH_CUES = re.compile(
    r"\b("
    r"depress(ed|ion)|anxious|anxiety|panic|panic\s+attack|overthink(ing)?|"
    r"can't\s+sleep|cannot\s+sleep|insomnia|lonely|hopeless|worthless|"
    r"cry(ing)?|burnout|overwhelmed|numb|empty|no\s+motivation"
    r")\b",
    re.I,
)

GREET_CUES = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening))\b[^\w]*$",
    re.I,
)

# ------------------------------------------------

PROMPT = (
    "You are a router. Read the user's line and choose exactly ONE label:\n"
    "G = greeting (short hello/intro only)\n"
    "M = mh_support (user describes feelings/symptoms/coping needs)\n"
    "I = info_definition (asks for definitions/explanations, e.g., 'What is depression?')\n"
    "O = other (anything else)\n\n"
    "Reply with ONLY the single letter: G or M or I or O.\n\n"
    "Text: {text}\n"
    "Label:"
)

@dataclass
class RouteDecision:
    route: str
    confidence: float
    label: str

_LABEL_MAP = {"G": "greeting", "M": "mh_support", "I": "info_definition", "O": "other"}

class LLMRouter:
    def __init__(self):
        self.client = model_manager.get_router_client()
        logger.info("[router] Router using model: %s", self.client.model_name)

    async def classify(self, text: str) -> RouteDecision:
        t = (text or "").strip()
        # 1) 先問 LLM（單字母）
        try:
            resp = await self.client.complete(
                PROMPT.format(text=t),
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=4,
                max_time=4.0,
            )
            label = (resp or "").strip().upper()[:1]
            if label not in _LABEL_MAP:
                label = "O"
        except Exception as e:
            logger.warning("Router LLM failed (%s), fallback to heuristic only.", e)
            label = "O"

        # 2) 啟發式覆寫（窄而準）
        #   * 先定義/解釋 → I
        if DEFN_CUES.search(t):
            label = "I"
        #   * 強 MH 描述 且不是定義題 → M
        elif MH_CUES.search(t):
            label = "M"
        #   * 純招呼（極短）→ G
        elif GREET_CUES.match(t):
            label = "G"

        route = _LABEL_MAP[label]
        # 粗估 confidence：啟發式命中就給 0.9，否則 0.5
        conf = 0.9 if (DEFN_CUES.search(t) or MH_CUES.search(t) or GREET_CUES.match(t)) else 0.5
        logger.info("Router classified '%s' -> %s (label=%s, conf=%.2f)", t[:30], route, label, conf)
        return RouteDecision(route=route, confidence=conf, label=label)
