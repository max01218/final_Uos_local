# app/orchestration/orchestrator.py
import re
import logging
from app.core.settings import settings
from app.clients.model_manager import model_manager
from app.orchestration.prompt_compiler import PromptCompiler
from app.orchestration.router_service import LLMRouter
from app.orchestration.judge_service import JudgeService
from app.orchestration.repair_service import RepairService
from app.orchestration.flow_service import GuidedFlowService
import asyncio

logger = logging.getLogger(__name__)

CRISIS_RE = re.compile(r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|homicide|kill (him|her|them))", re.I)

class Orchestrator:
    def __init__(self, rag=None, conversation_store=None):
        self.router = LLMRouter()
        self.compiler = PromptCompiler("app/prompts/registry.yaml")
        # Use shared model manager
        self.main = model_manager.get_main_client()
        self.judge = JudgeService(self.main)
        self.repairer = RepairService(self.main)
        self.rag = rag
        # Initialize flow service with conversation store
        if conversation_store:
            self.flow = GuidedFlowService(conversation_store, self.compiler, self.main)
        else:
            self.flow = None

    async def generate(self, *, question: str, history: str, tone: str = "balanced", session_id: str = None):
        # Crisis hard gate (regex only for safety)
        if CRISIS_RE.search(question):
            route = "crisis"
            route_score = 1.0
            logger.warning(f"Crisis detected: {question[:50]}...")
        else:
            # Stage-1: Router LLM classification
            decision = await self.router.classify(question)
            route = decision.route
            route_score = getattr(decision, 'score', getattr(decision, 'confidence', None))

        # Special handling for mh_support route with guided flow
        if route == "mh_support" and self.flow:
            return await self._handle_guided_flow(
                question=question, 
                history=history, 
                session_id=session_id
            )

        # RAG context for relevant routes
        context = ""
        if route in ("info_definition", "mh_support") and self.rag:
            try:
                if hasattr(self.rag, 'retrieve'):
                    docs = await self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
                else:
                    # Synchronous fallback
                    docs = self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                context = ""

        # Stage-2: Compile route-specific prompt 
        prompt = self.compiler.compile(
            route=route, 
            question=question, 
            history=history, 
            context=context, 
            tone=tone
        )
        
        # Stage-2: Generator LLM produces final response
        logger.info(f"Stage-2 Generator LLM generating for route: {route} (score: {route_score})")
        raw = await self.main.complete(prompt)
        
        # For simple routes like greeting, skip judge/repair to improve speed and quality
        if route in ("greeting", "small_talk"):
            logger.info(f"Skipping judge/repair for simple route: {route}")
            return raw.strip(), {"route": route, "route_score": route_score, "repaired": False, "fast_path": True}
        
        # Judge and repair for complex routes only
        spec = self.compiler.routes[route]
        constraints = self.compiler._join_constraints(spec.get("constraints", []))
        contract = self.compiler.contracts[spec["output_contract"]]
        
        ok = await self.judge.pass_fail(contract, constraints, question, raw)
        if not ok:
            logger.info(f"Response failed judge, attempting repair for route: {route}")
            try:
                fixed = await self.repairer.repair(contract, constraints, question, raw)
                return fixed.strip(), {"route": route, "route_score": route_score, "repaired": True}
            except Exception as e:
                logger.warning(f"Repair failed: {e}, using original")
                return raw.strip(), {"route": route, "route_score": route_score, "repaired": False, "repair_error": str(e)}
        
        return raw.strip(), {"route": route, "route_score": route_score, "repaired": False}

    async def _handle_guided_flow(self, *, question: str, history: str, session_id: str = None):
        """Handle guided flow for mh_support route"""
        logger.info(f"Handling guided flow for mh_support route")
        
        # Plan if needed (first time or flow not active)
        plan_json = await self.flow.plan_if_needed(
            question=question, 
            history=history, 
            session_id=session_id
        )

        # Get RAG context
        context = ""
        if self.rag:
            try:
                if hasattr(self.rag, 'retrieve'):
                    docs = await self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
                else:
                    docs = self.rag.retrieve(question, k=3)
                    context = self.rag.build_context(docs, max_docs=2)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                context = ""

        # Generate next turn response with hedged strategy (main vs fast)
        raw = await self._hedged_turn_generate(
            question=question,
            history=history,
            context=context,
            plan_json=plan_json,
            session_id=session_id
        )

        # Judge and repair for flow responses (structure + tone)
        spec = self.compiler.routes["mh_support"]
        constraints = self.compiler._join_constraints(spec.get("constraints", []))
        contract = self.compiler.contracts[spec["output_contract"]]
        tone_block = self.compiler.get_tone_block("balanced")
        banned = getattr(self.compiler, "banned_phrases", [])
        # quick-check before LLM judge
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
                return fast.strip(), {"route": "mh_support", "repaired": False, "flow_active": True, "quick_repair": True}
        
        # Time-box the judge call; on timeout, skip to repair fallback path via fast generation
        try:
            ok = await asyncio.wait_for(self.judge.pass_fail(contract, constraints, question, raw), timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("Judge timed out; using fast repair path")
            ok = False
        if not ok:
            logger.info(f"Flow response failed judge, attempting repair")
            try:
                # Time-box repair; if it times out or fails, regenerate via fast prompt as resilient fallback
                fixed = await asyncio.wait_for(self.repairer.repair(contract, constraints, question, raw), timeout=10.0)
                raw = fixed
            except Exception as e:
                logger.warning(f"Repair failed or timed out: {e}; regenerating via fast prompt")
                fast = await self._hedged_turn_generate(
                    question=question,
                    history=history,
                    context=context,
                    plan_json=plan_json,
                    session_id=session_id,
                )
                return fast.strip(), {"route": "mh_support", "repaired": False, "flow_active": True, "repair_error": str(e), "fast_regen": True}

        # Tone scoring and optional tone repair
        try:
            tone_ok = await asyncio.wait_for(self.judge.tone_score(tone_block, raw), timeout=6.0) >= 0.7
        except asyncio.TimeoutError:
            tone_ok = True  # skip tone repair on timeout for latency
        if not tone_ok:
            try:
                raw = await asyncio.wait_for(self.repairer.tone_repair(tone_block, contract, constraints, question, raw), timeout=12.0)
            except Exception as e:
                logger.warning(f"Tone repair failed: {e}")

        return raw.strip(), {"route": "mh_support", "repaired": not ok, "flow_active": True}

    async def _hedged_turn_generate(self, *, question: str, history: str, context: str, plan_json: str, session_id: str = None) -> str:
        """Race a main prompt vs a fast prompt; first result wins."""
        # Build prompts
        spec = self.compiler.routes["mh_support"]
        # main turn
        main_prompt = self.compiler.compile_flow_turn(
            route="mh_support", question=question, history=history, context=context,
            technique=self.flow.load_state(session_id).technique or "",
            step_index=self.flow.load_state(session_id).step_index,
            plan_json=plan_json or "{}",
            expected_question_type=self.flow.load_state(session_id).last_question_type,
        )
        # fast turn (short strict)
        fast_prompt = self.compiler.compile_flow_turn_fast(
            route="mh_support", question=question, history=history, context=context,
            technique=self.flow.load_state(session_id).technique or "",
            step_index=self.flow.load_state(session_id).step_index,
            plan_json=plan_json or "{}",
            expected_question_type=self.flow.load_state(session_id).last_question_type,
        )

        async def _gen_main():
            return await self.main.complete(main_prompt, max_time=30.0, max_new_tokens=None)

        async def _gen_fast():
            return await self.main.complete(fast_prompt, max_time=8.0, max_new_tokens=80)

        # Hedge with a soft delay for the fast path
        t_main = asyncio.create_task(_gen_main())
        await asyncio.sleep(2.0)
        t_fast = asyncio.create_task(_gen_fast())
        done, pending = await asyncio.wait({t_main, t_fast}, return_when=asyncio.FIRST_COMPLETED)
        winner = done.pop()
        for p in pending:
            p.cancel()
        try:
            return await winner
        except Exception:
            # Try the other task if winner failed
            for p in pending:
                try:
                    return await p
                except Exception:
                    pass
            raise
