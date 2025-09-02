# app/orchestration/orchestrator.py
import re
import logging
import asyncio

from app.clients.model_manager import model_manager
from app.orchestration.prompt_compiler import PromptCompiler
from app.orchestration.router_service import LLMRouter
from app.orchestration.judge_service import JudgeService
from app.orchestration.repair_service import RepairService
from app.orchestration.flow_service import GuidedFlowService

logger = logging.getLogger(__name__)

CRISIS_RE = re.compile(
    r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|i want to die|想死|自殺|傷害自己)",
    re.I,
)

CRISIS_TEMPLATE = (
    "I'm really sorry you're feeling this way, and I'm here with you. "
    "If you are in immediate danger or feel you might act on these thoughts, please call your local emergency number now.\n"
    "You can also talk to someone right away:\n"
    "- UK: Samaritans 116 123 (24/7)\n"
    "- US: 988 Suicide & Crisis Lifeline (24/7)\n"
    "- EU: 112 (general emergency)\n"
    "- Taiwan: 1925 Lifeline (24/7)\n"
    "If you can, consider telling a trusted person nearby. "
    "Would you like help finding the right number or planning one small step for safety tonight?"
)

_LABEL_LINE = re.compile(r"^\s*(E|S|Q)\s*:\s*", re.I)
_ROLE_HEAD = re.compile(r"(?i)\b(system|assistant|human|message|user)\b\s*[:：]\s*")
_CODE_FENCE = re.compile(r"(?is)`{3}.*?`{3}")
_ROLE_PREFIX = re.compile(r"^\s*(system|assistant|user|human|message)\s*[:：]\s*", re.I)

TONE_HINT = {
    "balanced": "Use a balanced, calm and supportive tone.",
    "warm": "Use a warmer, gentler, more encouraging tone.",
    "direct": "Use a concise, straightforward, no-fluff professional tone.",
}
def _tone_hint(tone: str) -> str:
    return TONE_HINT.get((tone or "balanced").lower(), TONE_HINT["balanced"])

def _strip_labels(text: str) -> str:
    if not text:
        return text
    text = _CODE_FENCE.sub("", text)
    out_lines = []
    for ln in text.splitlines():
        ln = _ROLE_HEAD.sub("", ln).strip()
        ln = _LABEL_LINE.sub("", ln).strip()
        if ln:
            out_lines.append(ln)
    import re as _re
    out = " ".join(out_lines).strip()
    return _re.sub(r"\s{2,}", " ", out)

def _clean_roles(text: str) -> str:
    lines = [_ROLE_PREFIX.sub("", ln).strip() for ln in (text or "").splitlines()]
    return " ".join([ln for ln in lines if ln]).strip()

def _sentences(text: str):
    import re as _re
    s = _re.split(r"(?<=[.!?])\s+", text.strip())
    return [x.strip() for x in s if x.strip()]

def _dedupe_sentences(text: str) -> str:
    seen = set()
    out = []
    for s in _sentences(text):
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return " ".join(out)

def _clip_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".,;:!?")

def _remove_qna_noise(text: str) -> str:
    bad_starts = (
        "human:", "assistant:", "system:", "user:", "message:",
        "can you", "how do i", "how can i", "could you", "would you",
        "answer:", "question:", "repaired reply:", "natural chat",
    )
    kept = []
    for s in _sentences(text):
        low = s.lower().strip()
        if any(low.startswith(b) for b in bad_starts):
            continue
        if " answer:" in low or " question:" in low:
            continue
        kept.append(s)
    return " ".join(kept)

class Orchestrator:
    def __init__(self, rag=None, conversation_store=None):
        self.router = LLMRouter()
        self.compiler = PromptCompiler("app/prompts/registry.yaml")
        self.main = model_manager.get_main_client()
        self.judge = JudgeService(self.main)
        self.repairer = RepairService(self.main)
        self.rag = rag
        self.flow = GuidedFlowService(conversation_store, self.compiler, self.main) if conversation_store else None

    async def _gen_greeting(self, user_text: str, tone: str = "balanced") -> str:
        prompt = (
            "Write a brief greeting to the user based on the line below.\n"
            f"{_tone_hint(tone)}\n"
            "Rules:\n"
            "- Max 30 words.\n"
            "- No lists, no labels, no emojis, no jokes, no new topics.\n"
            "- The only allowed question is 'How can I help today?'\n\n"
            f"User said: {user_text}\n"
            "Reply:"
        )
        resp = await self.main.complete(prompt, temperature=0.3, top_p=0.9, max_new_tokens=50, max_time=5.0)
        out = _strip_labels(resp or "")
        return "How can I help today?" if "?" in out else (out[:120].strip() or "Hi, how can I help today?")

    async def _gen_info_definition(self, question: str, context: str, tone: str = "balanced") -> str:
        prompt = (
            "Provide a concise, plain-English explanation to the user's question.\n"
            f"{_tone_hint(tone)}\n"
            "Rules:\n"
            "- 2 to 4 sentences.\n"
            "- No empathy lines, no therapeutic steps, no lists, no labels.\n"
            "- Do not include any role prefixes like 'Human:' or 'Assistant:'.\n"
            "- Avoid any Q&A scaffolding or self-questions.\n\n"
            f"Question: {question}\n"
        )
        if context:
            prompt += f"Helpful context:\n{context}\n"
        prompt += "\nAnswer:"

        raw = await self.main.complete(prompt, temperature=0.2, top_p=0.95, max_new_tokens=180, max_time=8.0)
        cleaned = _clean_roles(raw or "")
        cleaned = _remove_qna_noise(_dedupe_sentences(cleaned))
        sents = _sentences(cleaned)
        if len(sents) > 4:
            cleaned = " ".join(sents[:4])
        cleaned = _clip_words(cleaned, 80)
        return cleaned.strip()

    async def generate(self, *, question: str, history: str, tone: str = "balanced", session_id: str = None):
        if CRISIS_RE.search(question or ""):
            return CRISIS_TEMPLATE, {"route": "crisis", "route_score": 1.0, "naturalized": True}

        decision = await self.router.classify(question or "")
        route = decision.route
        route_score = decision.confidence

        if route == "mh_support" and self.flow:
            final_raw, meta = await self._handle_guided_flow(
                question=question, history=history, session_id=session_id, tone=tone
            )
            return (final_raw or "").strip(), {**(meta or {}), "route": route, "route_score": route_score}

        if route == "greeting":
            final = await self._gen_greeting(question, tone)
            return final, {"route": route, "route_score": route_score, "naturalized": True}

        if route == "info_definition":
            context = ""
            if self.rag:
                try:
                    if hasattr(self.rag, "retrieve"):
                        docs = await self.rag.retrieve(question, k=3)
                        context = self.rag.build_context(docs, max_docs=2)
                    else:
                        docs = self.rag.retrieve(question, k=3)
                        context = self.rag.build_context(docs, max_docs=2)
                except Exception as e:
                    logger.warning("RAG retrieval failed: %s", e)
            final = await self._gen_info_definition(question, context, tone)
            return final, {"route": route, "route_score": route_score, "naturalized": True}

        prompt = self.compiler.compile(route=route, question=question, history=history, context="", tone=tone)
        prompt = f"{prompt}\n\nTone guideline:\n{_tone_hint(tone)}"
        raw = await self.main.complete(prompt)
        final = _strip_labels((raw or "").strip())
        return final, {"route": route, "route_score": route_score, "naturalized": True}

    async def _handle_guided_flow(self, *, question: str, history: str, session_id: str = None, tone: str = "balanced"):
        logger.info("Handling guided flow for mh_support route")
        plan_json = await self.flow.plan_if_needed(
            question=question, history=history, session_id=session_id
        )

        context = ""
        if self.rag:
            try:
                if hasattr(self.rag, "retrieve"):
                    docs = await self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
                else:
                    docs = self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)

        raw = await self._hedged_turn_generate(
            question=question, history=history, context=context, plan_json=plan_json or "{}", session_id=session_id, tone=tone
        )

        try:
            spec = self.compiler.routes["mh_support"]
            constraints = self.compiler._join_constraints(spec.get("constraints", []))
            contract = self.compiler.contracts[spec.get("output_contract", "esq_three_lines")]
        except Exception:
            return (raw or "").strip(), {"route": "mh_support", "repaired": False, "flow_active": True}

        ok = True
        try:
            ok = await asyncio.wait_for(self.judge.pass_fail(contract, constraints, question, raw), timeout=8.0)
        except asyncio.TimeoutError:
            ok = True
        except Exception:
            ok = True

        if not ok:
            try:
                raw = await asyncio.wait_for(self.repairer.repair(contract, constraints, question, raw), timeout=12.0)
            except Exception:
                pass

        return (raw or "").strip(), {"route": "mh_support", "repaired": not ok, "flow_active": True}

    async def _hedged_turn_generate(self, *, question: str, history: str, context: str, plan_json: str, session_id: str = None, tone: str = "balanced") -> str:
        s = self.flow.load_state(session_id)
        main_prompt = self.compiler.compile_flow_turn(
            route="mh_support", question=question, history=history, context=context,
            technique=s.technique or "", step_index=s.step_index,
            plan_json=plan_json or "{}", expected_question_type=s.last_question_type,
        )
        fast_prompt = self.compiler.compile_flow_turn_fast(
            route="mh_support", question=question, history=history, context=context,
            technique=s.technique or "", step_index=s.step_index,
            plan_json=plan_json or "{}", expected_question_type=s.last_question_type,
        )

        main_prompt = f"{main_prompt}\n\nTone guideline:\n{_tone_hint(tone)}"
        fast_prompt = f"{fast_prompt}\n\nTone guideline:\n{_tone_hint(tone)}"

        async def _gen_main():
            return await self.main.complete(main_prompt, max_time=30.0, max_new_tokens=None)

        async def _gen_fast():
            return await self.main.complete(fast_prompt, max_time=8.0, max_new_tokens=80)

        t_main = asyncio.create_task(_gen_main())
        await asyncio.sleep(2.0)
        t_fast = asyncio.create_task(_gen_fast())
        done, pending = await asyncio.wait({t_main, t_fast}, return_when=asyncio.FIRST_COMPLETED)
        winner = next(iter(done))
        for p in pending:
            p.cancel()
        try:
            return await winner
        except Exception:
            for p in pending:
                try:
                    return await p
                except Exception:
                    pass
            return ""
