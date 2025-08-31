# app/orchestration/judge_service.py
from app.clients.llm_client import LLMClient
import re
import json
from typing import Tuple

JUDGE_PROMPT = """Evaluate the assistant reply against the Output Contract and Constraints.
Return EXACTLY one word: PASS or FAIL.

Contract:
{contract}

Constraints:
{constraints}

User:
{question}

Assistant Reply:
{assistant_raw}

Evaluation:"""

TONE_JUDGE_PROMPT = """Given the Tone Profile and the assistant reply, score adherence from 0 to 1.
Return STRICT JSON: {"score": 0.0-1.0, "violations": ["..."]}

Tone Profile:
{tone_profile}

Assistant:
{assistant_raw}
"""

class JudgeService:
    def __init__(self, client: LLMClient): 
        self.client = client
        
    async def pass_fail(self, contract: str, constraints: str, question: str, assistant_raw: str) -> bool:
        out = await self.client.complete(JUDGE_PROMPT.format(
            contract=contract, 
            constraints=constraints, 
            question=question, 
            assistant_raw=assistant_raw
        ))
        return "PASS" in out.upper()

    async def tone_score(self, tone_profile: str, assistant_raw: str) -> float:
        out = await self.client.complete(TONE_JUDGE_PROMPT.format(
            tone_profile=tone_profile,
            assistant_raw=assistant_raw
        ), max_time=6.0, max_new_tokens=80)
        m = re.search(r"\{.*?\}", out, re.S)
        try:
            data = json.loads(m.group(0)) if m else {"score": 0.0}
        except Exception:
            data = {"score": 0.0}
        try:
            return float(data.get("score", 0.0))
        except Exception:
            return 0.0

    def quick_check(self, *, assistant_raw: str, banned_phrases: list, recent_es: list) -> Tuple[bool, str]:
        """Lightweight local checks before calling LLM judge.
        - Ensure exactly three lines with E:/S:/Q:
        - No banned phrases
        - E trigram not repeating recent E lines
        """
        text = (assistant_raw or "").strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) != 3 or not lines[0].startswith("E:") or not lines[1].startswith("S:") or not lines[2].startswith("Q:"):
            return False, "structure"
        low = text.lower()
        for bp in banned_phrases or []:
            if bp and bp.lower() in low:
                return False, "banned"
        # trigram repetition on E line
        if recent_es:
            import itertools
            def trigrams(s):
                toks = s.split()
                return set(" ".join(toks[i:i+3]).lower() for i in range(max(0, len(toks)-2)))
            e_tri = trigrams(lines[0][2:].strip())
            for prev in recent_es[-3:]:
                if trigrams(prev).intersection(e_tri):
                    return False, "trigram"
        return True, "ok"
