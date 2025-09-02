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
- Keep 1-2 concrete, low-burden suggestions as plain sentences.
- Ask EXACTLY ONE short question at the end.
- Vary sentence length with gentle connectors ("if it's okay", "we can start small").
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

CRISIS_RE = re.compile(
    r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|homicide|kill (him|her|them))",
    re.I,
)

def _short_surface_enforce(t: str, max_words: int = 35) -> str:
    if not t:
        return t
    # drop code fences & system-ish headers
    t = re.sub(r"(?is)`{3}.*?`{3}", "", t)
    t = re.sub(r"(?i)\b(system|assistant|human|message)\b\s*[:：]\s*", "", t)
    t = t.strip()
    # first non-empty line
    for line in t.splitlines():
        line = line.strip()
        if line:
            t = line
            break
    # clip to max words
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(".,;:! ")
    # ensure one short question ending
    if not t.endswith("?"):
        t = t.rstrip(".! ") + " - is that okay?"
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
            return (resp or draft).strip()
        except Exception as e:
            logger.warning(f"Naturalizer failed: {e}")
            return draft

    async def _naturalize_with_style(self, text: str, style: str = "default") -> str:
        draft = (text or "").strip()
        if not draft:
            return draft
        try:
            prompt = NATURALIZER_GREETING.format(draft=draft) if style == "greeting" else NATURALIZER_PROMPT.format(draft=draft)
            logger.info("Naturalizer: rewriting draft to conversational surface...")
            resp = await self.main.complete(prompt, temperature=0.6, top_p=0.9, max_new_tokens=200, max_time=8.0)
            out = (resp or draft).strip()
            return _short_surface_enforce(out, max_words=35) if style == "greeting" else out
        except Exception as e:
            logger.warning(f"Naturalizer failed: {e}")
            return _short_surface_enforce(draft, max_words=35) if style == "greeting" else draft

    async def generate(self, *, question: str, history: str, tone: str = "balanced", session_id: str = None):
        # Stage 0: crisis hard gate
        if CRISIS_RE.search(question or ""):
            route = "crisis"
            route_score = 1.0
            logger.warning(f"Crisis detected: {(question or '')[:50]}...")
        else:
            # Stage 1: router classification (always run)
            decision = await self.router.classify(question or "")
            route = decision.route
            route_score = getattr(decision, "score", getattr(decision, "confidence", None))

        # Guided flow for mh_support
        if route == "mh_support" and self.flow:
            final_raw, meta = await self._handle_guided_flow(
                question=question, history=history, session_id=session_id
            )
            final = await self._naturalize(final_raw)
            meta = {**(meta or {}), "route_score": route_score, "naturalized": True}
            return final, meta

        # Optional RAG for specific routes
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

        # Stage 2: compile prompt and generate
        prompt = self.compiler.compile(route=route, question=question, history=history, context=context, tone=tone)
        logger.info(f"Stage-2 Generator LLM generating for route: {route} (score: {route_score})")
        raw = await self.main.complete(prompt)

        # Simple routes: short naturalized greeting/small talk
        if route in ("greeting", "small_talk"):
            logger.info(f"Skipping judge/repair for simple route: {route}")
            final = await self._naturalize_with_style(raw.strip(), style="greeting")
            return final, {
                "route": route,
                "route_score": route_score,
                "repaired": False,
                "fast_path": True,
                "naturalized": True,
            }

        # Complex routes: judge & repair
        spec = self.compiler.routes[route]
        constraints = self.compiler._join_constraints(spec.get("constraints", []))
        contract = self.compiler.contracts[spec["output_contract"]]

        ok = await self.judge.pass_fail(contract, constraints, question, raw)
        if not ok:
            logger.info(f"Response failed judge, attempting repair for route: {route}")
            try:
                fixed = await self.repairer.repair(contract, constraints, question, raw)
                final = await self._naturalize(fixed.strip())
                return final, {
                    "route": route,
                    "route_score": route_score,
                    "repaired": True,
                    "naturalized": True,
                }
            except Exception as e:
                logger.warning(f"Repair failed: {e}, using original")
                final = await self._naturalize(raw.strip())
                return final, {
                    "route": route,
                    "route_score": route_score,
                    "repaired": False,
                    "repair_error": str(e),
                    "naturalized": True,
                }

        final = await self._naturalize(raw.strip())
        return final, {"route": route, "route_score": route_score, "repaired": False, "naturalized": True}

    async def _handle_guided_flow(self, *, question: str, history: str, session_id: str = None):
        logger.info("Handling guided flow for mh_support route")

        # Plan if needed
        plan_json = await self.flow.plan_if_needed(question=question, history=history, session_id=session_id)

        # Optional RAG
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

        # Hedge: main vs fast
        raw = await self._hedged_turn_generate(
            question=question, history=history, context=context, plan_json=plan_json, session_id=session_id
        )

        # Judge & repair tuned for flow
        spec = self.compiler.routes["mh_support"]
        constraints = self.compiler._join_constraints(spec.get("constraints", []))
        contract = self.compiler.contracts[spec["output_contract"]]
        tone_block = self.compiler.get_tone_block("balanced")
        banned = getattr(self.compiler, "banned_phrases", [])

        # Quick check before LLM judge
        e_history = []
        try:
            recent = self.flow.cs.get_recent_messages(3)
            for m in recent:
                if m.get("role") == "assistant":
                    txt = m.get("content", "")
                    if txt.startswith("E:"):
                        e_history.append(txt)
        except Exception:
            e_history = []

        qc_ok, qc_reason = self.judge.quick_check(assistant_raw=raw, banned_phrases=banned, recent_es=e_history)
        if not qc_ok:
            logger.info(f"Quick-check failed ({qc_reason}); attempting repair")
            try:
                fixed = await asyncio.wait_for(self.repairer.repair(contract, constraints, question, raw), timeout=10.0)
                raw = fixed
            except Exception as e:
                logger.warning(f"Quick repair failed: {e}; using fast prompt")
                fast = await self._hedged_turn_generate(
                    question=question, history=history, context=context, plan_json=plan_json, session_id=session_id
                )
                fast = await self._naturalize(fast.strip())
                return fast, {"route": "mh_support", "repaired": False, "flow_active": True, "quick_repair": True, "naturalized": True}

        try:
            ok = await asyncio.wait_for(self.judge.pass_fail(contract, constraints, question, raw), timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("Judge timed out; using fast repair path")
            ok = False

        if not ok:
            logger.info("Flow response failed judge, attempting repair")
            try:
                fixed = await asyncio.wait_for(self.repairer.repair(contract, constraints, question, raw), timeout=10.0)
                raw = fixed
            except Exception as e:
                logger.warning(f"Repair failed or timed out: {e}; regenerating via fast prompt")
                fast = await self._hedged_turn_generate(
                    question=question, history=history, context=context, plan_json=plan_json, session_id=session_id
                )
                fast = await self._naturalize(fast.strip())
                return fast, {"route": "mh_support", "repaired": False, "flow_active": True, "repair_error": str(e), "fast_regen": True, "naturalized": True}

        # Tone scoring and optional tone repair
        try:
            tone_ok = await asyncio.wait_for(self.judge.tone_score(tone_block, raw), timeout=6.0) >= 0.7
        except asyncio.TimeoutError:
            tone_ok = True

        if not tone_ok:
            try:
                raw = await asyncio.wait_for(self.repairer.tone_repair(tone_block, contract, constraints, question, raw), timeout=12.0)
            except Exception as e:
                logger.warning(f"Tone repair failed: {e}")

        final = await self._naturalize(raw.strip())
        return final, {"route": "mh_support", "repaired": not ok, "flow_active": True, "naturalized": True}

    async def _hedged_turn_generate(self, *, question: str, history: str, context: str, plan_json: str, session_id: str = None) -> str:
        """Race a main prompt vs a fast prompt; first result wins."""
        state = self.flow.load_state(session_id) if self.flow else None
        technique = state.technique if state else ""
        step_index = state.step_index if state else 0
        expected_q = state.last_question_type if state else None

        main_prompt = self.compiler.compile_flow_turn(
            route="mh_support", question=question, history=history, context=context,
            technique=technique, step_index=step_index, plan_json=plan_json or "{}", expected_question_type=expected_q
        )
        fast_prompt = self.compiler.compile_flow_turn_fast(
            route="mh_support", question=question, history=history, context=context,
            technique=technique, step_index=step_index, plan_json=plan_json or "{}", expected_question_type=expected_q
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
            raise
