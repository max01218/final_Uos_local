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

NATURALIZER_PROMPT = """You will be given a DRAFT reply for a chat. It may include labels like "E:", "S:", "Q:".
Rewrite it as ONE natural chat message in warm, professional English.

Rules:
- Do NOT show any labels (E:, S:, Q:) or bullets/numbers.
- Start with a brief human acknowledgement (1 short sentence).
- Keep 1–2 concrete, low-burden suggestions as plain sentences.
- Ask EXACTLY ONE short question at the end.
- Under 120 words.

DRAFT:
{draft}

Natural chat message:"""

NATURALIZER_GREETING = """You will be given a DRAFT greeting reply.
Rewrite it as ONE short, natural chat message.

Rules:
- Max 35 words total.
- Warm, concise, no lists, no labels.
- End with exactly ONE short question.

DRAFT:
{draft}

Message:"""

CRISIS_RE = re.compile(r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|homicide|kill (him|her|them))", re.I)

def _strip_labels(text: str) -> str:
    import re
    _LABEL_LINE = re.compile(r"^\s*(E|S|Q)\s*:\s*", re.I)
    _ROLE_HEAD = re.compile(r"(?i)\b(system|assistant|human|message)\b\s*[:：]\s*")
    _CODE_FENCE = re.compile(r"(?is)`{3}.*?`{3}")
    if not text:
        return text
    text = _CODE_FENCE.sub("", text)
    out_lines = []
    for ln in text.splitlines():
        ln = _ROLE_HEAD.sub("", ln).strip()
        ln = _LABEL_LINE.sub("", ln).strip()
        if ln:
            out_lines.append(ln)
    out = " ".join(out_lines).strip()
    return re.sub(r"\s{2,}", " ", out)

def _short_surface_enforce(t: str, max_words: int = 35, ensure_question: bool = True) -> str:
    t = _strip_labels(t or "")
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(".,;:! ")
    if ensure_question and not t.endswith("?"):
        t = t.rstrip(".! ") + " — is that okay?"
    return t

class Orchestrator:
    def __init__(self, rag=None, conversation_store=None):
        self.router = LLMRouter()
        self.compiler = PromptCompiler("app/prompts/registry.yaml")
        self.main = model_manager.get_main_client()
        self.judge = JudgeService(self.main)
        self.repairer = RepairService(self.main)
        self.rag = rag
        self.flow = GuidedFlowService(conversation_store, self.compiler, self.main) if conversation_store else None

    async def _naturalize(self, text: str) -> str:
        draft = (text or "").strip()
        if not draft:
            return draft
        try:
            logger.info("Naturalizer: rewriting draft to conversational surface...")
            resp = await self.main.complete(
                NATURALIZER_PROMPT.format(draft=draft),
                temperature=0.6, top_p=0.9, max_new_tokens=200, max_time=8.0
            )
            return _strip_labels(resp or draft)
        except Exception as e:
            logger.warning(f"Naturalizer failed: {e}")
            return _strip_labels(draft)

    async def _naturalize_with_style(self, text: str, style: str = "greeting") -> str:
        draft = (text or "").strip()
        if not draft:
            return draft
        try:
            prompt = NATURALIZER_GREETING.format(draft=draft)
            logger.info("Naturalizer: rewriting draft to conversational surface...")
            resp = await self.main.complete(prompt, temperature=0.6, top_p=0.9, max_new_tokens=200, max_time=8.0)
            out = _strip_labels(resp or draft)
            return _short_surface_enforce(out, 35, True)
        except Exception as e:
            logger.warning(f"Naturalizer failed: {e}")
            out = _strip_labels(draft)
            return _short_surface_enforce(out, 35, True)

    async def generate(self, *, question: str, history: str, tone: str = "balanced", session_id: str = None):
        if CRISIS_RE.search(question or ""):
            route, route_score = "crisis", 1.0
        else:
            decision = await self.router.classify(question or "")
            route = decision.route
            route_score = getattr(decision, "score", getattr(decision, "confidence", None))

        if route == "mh_support" and self.flow:
            final_raw, meta = await self._handle_guided_flow(question=question, history=history, session_id=session_id)
            meta = {**(meta or {}), "route_score": route_score, "naturalized": False}
            return (final_raw or "").strip(), meta

        context = ""
        if route in ("info_definition", "mh_support") and self.rag:
            try:
                if hasattr(self.rag, "retrieve"):
                    docs = await self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
                else:
                    docs = self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                context = ""

        prompt = self.compiler.compile(route=route, question=question, history=history, context=context, tone=tone)
        logger.info(f"Stage-2 Generator LLM generating for route: {route} (score: {route_score})")
        raw = await self.main.complete(prompt)

        if route in ("greeting", "small_talk"):
            logger.info(f"Skipping judge/repair for simple route: {route}")
            final = await self._naturalize_with_style(raw.strip(), style="greeting")
            return final, {"route": route, "route_score": route_score, "repaired": False, "fast_path": True, "naturalized": True}

        try:
            spec = self.compiler.routes[route]
        except Exception as e:
            logger.warning(f"Compiler spec missing for route '{route}': {e}; returning raw")
            return (raw or "").strip(), {"route": route, "route_score": route_score, "repaired": False, "naturalized": False}

        constraints = self.compiler._join_constraints(spec.get("constraints", []))
        contract_key = spec.get("output_contract", "")
        contract = self.compiler.contracts.get(contract_key, "")
        requires_esq = (contract_key == "esq_three_lines")

        ok = await self.judge.pass_fail(contract, constraints, question, raw)
        if not ok:
            logger.info(f"Response failed judge, attempting repair for route: {route}")
            try:
                fixed = await self.repairer.repair(contract, constraints, question, raw)
                if requires_esq:
                    return (fixed or "").strip(), {"route": route, "route_score": route_score, "repaired": True, "naturalized": False}
                final = await self._naturalize((fixed or "").strip())
                return final, {"route": route, "route_score": route_score, "repaired": True, "naturalized": True}
            except Exception as e:
                logger.warning(f"Repair failed: {e}, using original")
                if requires_esq:
                    return (raw or "").strip(), {"route": route, "route_score": route_score, "repaired": False, "repair_error": str(e), "naturalized": False}
                final = await self._naturalize((raw or "").strip())
                return final, {"route": route, "route_score": route_score, "repaired": False, "repair_error": str(e), "naturalized": True}

        if requires_esq:
            return (raw or "").strip(), {"route": route, "route_score": route_score, "repaired": False, "naturalized": False}
        final = await self._naturalize((raw or "").strip())
        return final, {"route": route, "route_score": route_score, "repaired": False, "naturalized": True}

    async def _handle_guided_flow(self, *, question: str, history: str, session_id: str = None):
        logger.info("Handling guided flow for mh_support route")
        plan_json = await self.flow.plan_if_needed(question=question, history=history, session_id=session_id)

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
                logger.warning(f"RAG retrieval failed: {e}")
                context = ""

        raw = await self._hedged_turn_generate(
            question=question, history=history, context=context, plan_json=plan_json or "{}", session_id=session_id
        )

        try:
            spec = self.compiler.routes["mh_support"]
            constraints = self.compiler._join_constraints(spec.get("constraints", []))
            contract = self.compiler.contracts[spec.get("output_contract", "esq_three_lines")]
        except Exception as e:
            logger.warning(f"Compiler spec missing for mh_support: {e}")
            return (raw or "").strip(), {"route": "mh_support", "repaired": False, "flow_active": True}

        ok = True
        try:
            ok = await asyncio.wait_for(self.judge.pass_fail(contract, constraints, question, raw), timeout=8.0)
        except asyncio.TimeoutError:
            ok = True
        except Exception as e:
            logger.warning(f"judge.pass_fail error: {e}")
            ok = True

        if not ok:
            try:
                raw = await asyncio.wait_for(self.repairer.repair(contract, constraints, question, raw), timeout=12.0)
            except Exception as e:
                logger.warning(f"Flow repair failed: {e}")

        return (raw or "").strip(), {"route": "mh_support", "repaired": not ok, "flow_active": True}

    async def _hedged_turn_generate(self, *, question: str, history: str, context: str, plan_json: str, session_id: str = None) -> str:
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
