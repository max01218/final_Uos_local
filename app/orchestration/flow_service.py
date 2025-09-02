# app/orchestration/flow_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
import re
import logging

from app.services.memory_service import ConversationStore
from app.orchestration.prompt_compiler import PromptCompiler
from app.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)

DONE_RE = re.compile(r"\b(done|finished|complete(d)?|ok(ay)?|yes|continue|let'?s go|go ahead)\b", re.I)
RATING_RE = re.compile(r"\b(10|[0-9])\b")
STUCK_RE = re.compile(r"\b(difficult|hard|stuck|cannot|can'?t|unable|too much|struggle)\b", re.I)

ROLE_HEAD = re.compile(r"(?i)\b(system|assistant|human|message)\b\s*[:：]\s*")
CODE_FENCE = re.compile(r"(?is)`{3}.*?`{3}")
WS2 = re.compile(r"\s{2,}")

def _sid(session_id: Optional[str]) -> str:
    s = (session_id or "").strip()
    return s if s else "default"

@dataclass
class FlowState:
    active: bool = False
    technique: Optional[str] = None
    step_index: int = 0
    last_rating: Optional[int] = None
    last_question_type: str = "rating_0_10"
    plan_json: str = ""
    total_steps: int = 0

class GuidedFlowService:
    def __init__(self, cs: ConversationStore, compiler: PromptCompiler, llm: LLMClient):
        self.cs = cs
        self.compiler = compiler
        self.llm = llm

    def load_state(self, session_id: Optional[str] = None) -> FlowState:
        sid = _sid(session_id)
        return FlowState(
            active=bool(self.cs.get_flag(sid, "flow_active") or False),
            technique=self.cs.get_flag(sid, "technique"),
            step_index=int(self.cs.get_flag(sid, "step_index") or 0),
            last_rating=self.cs.get_flag(sid, "last_rating"),
            last_question_type=self.cs.get_flag(sid, "last_question_type") or "rating_0_10",
            plan_json=self.cs.get_flag(sid, "plan_json") or "",
            total_steps=int(self.cs.get_flag(sid, "total_steps") or 0),
        )

    def save_state(self, s: FlowState, session_id: Optional[str] = None) -> None:
        sid = _sid(session_id)
        self.cs.set_flag(sid, "flow_active", s.active)
        self.cs.set_flag(sid, "technique", s.technique)
        self.cs.set_flag(sid, "step_index", s.step_index)
        self.cs.set_flag(sid, "last_rating", s.last_rating)
        self.cs.set_flag(sid, "last_question_type", s.last_question_type)
        self.cs.set_flag(sid, "plan_json", s.plan_json)
        self.cs.set_flag(sid, "total_steps", s.total_steps)

    def _extract_rating(self, text: str) -> Optional[int]:
        m = RATING_RE.search(text or "")
        if not m:
            return None
        try:
            val = int(m.group(1))
            return val if 0 <= val <= 10 else None
        except Exception:
            return None

    def ingest_user_feedback(self, user_text: str) -> Dict[str, Any]:
        rating = self._extract_rating(user_text or "")
        cont = bool(DONE_RE.search(user_text or ""))
        stuck = bool(STUCK_RE.search(user_text or ""))
        return {"rating": rating, "continue": cont, "stuck": stuck}

    async def plan_if_needed(self, *, question: str, history: str, session_id: Optional[str] = None) -> str:
        s = self.load_state(session_id)
        if s.active and s.technique:
            return s.plan_json or ""

        prompt = self.compiler.compile_flow_plan(route="mh_support", question=question, history=history)
        plan_json = await self.llm.complete(prompt, max_time=20.0, max_new_tokens=200)

        try:
            technique = self.compiler.extract_technique(plan_json) or "Breathing"
        except Exception:
            logger.warning("extract_technique failed; using default 'Breathing'")
            technique = "Breathing"

        try:
            total_steps = int(self.compiler.extract_total_steps(plan_json) or 3)
        except Exception:
            logger.warning("extract_total_steps failed; using default 3")
            total_steps = 3

        s.active = True
        s.technique = technique
        s.step_index = 0
        s.last_question_type = "rating_0_10"
        s.plan_json = plan_json or ""
        s.total_steps = max(int(total_steps), 1)
        self.save_state(s, session_id)
        return s.plan_json

    async def next_turn(self, *, question: str, history: str, context: str, plan_json: Optional[str], session_id: Optional[str] = None) -> str:
        s = self.load_state(session_id)
        if s.total_steps <= 0:
            s.total_steps = 3
            self.save_state(s, session_id)
        if s.step_index >= s.total_steps:
            return await self._generate_wrap_up(question=question, history=history, technique=s.technique or "", session_id=session_id)

        prompt = self.compiler.compile_flow_turn(
            route="mh_support",
            question=question,
            history=history,
            context=context,
            technique=s.technique or "",
            step_index=int(s.step_index),
            plan_json=plan_json or s.plan_json or "{}",
            expected_question_type=s.last_question_type or "rating_0_10",
        )
        raw = await self.llm.complete(prompt)
        return self._clean_esq_response(raw)

    async def _generate_wrap_up(self, *, question: str, history: str, technique: str, session_id: Optional[str] = None) -> str:
        prompt = self.compiler.compile_flow_wrap_up(question=question, history=history, technique=technique or "")
        raw = await self.llm.complete(prompt)
        text = self._clean_esq_response(raw)
        self.save_state(FlowState(active=False), session_id)
        return text

    async def handle_adjustment(self, *, question: str, history: str, problem: str, session_id: Optional[str] = None) -> str:
        s = self.load_state(session_id)
        problem_safe = (problem or "").strip() or "The current step feels too difficult."
        prompt = self.compiler.compile_flow_adjust(question=question, history=history, technique=s.technique or "", problem=problem_safe)
        raw = await self.llm.complete(prompt)
        text = self._clean_esq_response(raw)
        s.last_question_type = "yes_no"
        self.save_state(s, session_id)
        return text

    def advance_or_adjust(self, user_text: str, session_id: Optional[str] = None) -> None:
        s = self.load_state(session_id)
        fb = self.ingest_user_feedback(user_text or "")
        if fb["rating"] is not None:
            s.last_rating = fb["rating"]
            if s.last_rating <= 3:
                s.step_index = max(0, s.step_index)
                s.last_question_type = "yes_no"
            else:
                s.step_index += 1
                s.last_question_type = "rating_0_10"
        elif fb["continue"]:
            s.step_index += 1
            s.last_question_type = "rating_0_10"
        elif fb["stuck"]:
            s.last_question_type = "yes_no"
        else:
            s.last_question_type = "yes_no"
        self.save_state(s, session_id)

    def complete_if_finished(self, session_id: Optional[str] = None, max_steps: int = 5) -> bool:
        s = self.load_state(session_id)
        return s.step_index >= max(s.total_steps or 0, max_steps)

    def is_flow_active(self, session_id: Optional[str] = None) -> bool:
        s = self.load_state(session_id)
        return bool(s.active)

    def _clean_esq_response(self, response: str) -> str:
        text = (response or "").strip()
        text = CODE_FENCE.sub("", text)
        text = ROLE_HEAD.sub("", text)
        text = WS2.sub(" ", text)

        e_match = re.search(r"(^|\n)\s*E:\s*(.+)", text, re.I)
        s_match = re.search(r"(^|\n)\s*S:\s*(.+)", text, re.I)
        q_match = re.search(r"(^|\n)\s*Q:\s*(.+)", text, re.I)

        def _pick(m) -> str:
            if not m:
                return ""
            seg = m.group(2).strip()
            seg = re.split(r"(^|\n)\s*[ESQ]\s*:\s*", seg, maxsplit=1, flags=re.I)[0].strip()
            return seg

        e = _pick(e_match)
        s = _pick(s_match)
        q = _pick(q_match)

        if not e:
            e = "I’m here with you; we can take this one step at a time."
        if not s:
            s = "Try a 60-second grounding: name 5 things you can see."
        if not q:
            q = "What feels most manageable to try next?"

        return f"E: {e}\nS: {s}\nQ: {q}"
