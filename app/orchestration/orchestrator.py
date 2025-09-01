# app/orchestration/orchestrator.py
import re
import logging
import asyncio

from app.core.settings import settings
from app.clients.model_manager import model_manager
from app.orchestration.prompt_compiler import PromptCompiler
from app.orchestration.router_service import LLMRouter
from app.orchestration.judge_service import JudgeService
from app.orchestration.repair_service import RepairService
from app.orchestration.flow_service import GuidedFlowService

logger = logging.getLogger(__name__)

# 將骨架式輸出（含 E:/S:/Q:）改寫為自然對話的提示
NATURALIZER_PROMPT = """You will be given a DRAFT reply for a chat. It may include labels like "E:", "S:", "Q:".
Rewrite it as ONE natural chat message in warm, professional English.

Rules:
- Do NOT show any labels (E:, S:, Q:) or bullets/numbers.
- Start with a brief human acknowledgement (1 short sentence).
- Keep 1–2 concrete, low-burden suggestions as plain sentences.
- Ask EXACTLY ONE short question at the end.
- Vary sentence length with gentle connectors ("if it's okay", "we can start small").
- Under 120 words.

DRAFT:
{draft}

Natural chat message:"""

CRISIS_RE = re.compile(
    r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|homicide|kill (him|her|them))",
    re.I,
)


class Orchestrator:
    def __init__(self, rag=None, conversation_store=None):
        self.router = LLMRouter()
        self.compiler = PromptCompiler("app/prompts/registry.yaml")
        # shared model（由 model_manager 決定具體 7B/3B）
        self.main = model_manager.get_main_client()
        self.judge = JudgeService(self.main)
        self.repairer = RepairService(self.main)
        self.rag = rag
        # Initialize flow service with conversation store
        if conversation_store:
            self.flow = GuidedFlowService(conversation_store, self.compiler, self.main)
        else:
            self.flow = None

    async def _naturalize(self, text: str) -> str:
        """把產生的草稿改寫成自然對話（失敗則回原文）"""
        draft = (text or "").strip()
        if not draft:
            return draft
        try:
            logger.info("Naturalizer: rewriting draft to conversational surface...")
            resp = await self.main.complete(
                NATURALIZER_PROMPT.format(draft=draft),
                temperature=0.6,
                max_new_tokens=200,
            )
            final = (resp or draft).strip()
            return final
        except Exception as e:
            logger.warning(f"Naturalizer failed: {e}")
            return draft

    async def generate(
        self, *, question: str, history: str, tone: str = "balanced", session_id: str = None
    ):
        # Stage-0: 危機字樣硬閘（regex 只做安全用）
        if CRISIS_RE.search(question or ""):
            route = "crisis"
            route_score = 1.0
            logger.warning(f"Crisis detected: {(question or '')[:50]}...")
        else:
            # Stage-1: 路由 LLM 分類（永遠先經過分類）
            decision = await self.router.classify(question or "")
            route = decision.route
            route_score = getattr(decision, "score", getattr(decision, "confidence", None))

        # mh_support → Guided Flow
        if route == "mh_support" and self.flow:
            final, meta = await self._handle_guided_flow(
                question=question, history=history, session_id=session_id
            )
            # flow 回傳內部多是骨架式輸出，統一自然化
            final = await self._naturalize(final)
            meta = {**(meta or {}), "route_score": route_score, "naturalized": True}
            return final, meta

        # Stage-1.5: 取 RAG context（僅在需要時）
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

        # Stage-2: 編譯 route 專屬提示
        prompt = self.compiler.compile(
            route=route, question=question, history=history, context=context, tone=tone
        )

        # Stage-2: 主要生成
        logger.info(
            f"Stage-2 Generator LLM generating for route: {route} (score: {route_score})"
        )
        raw = await self.main.complete(prompt)

        # 簡單路線 → 直接自然化後回傳（跳過 judge/repair 提升體感）
        if route in ("greeting", "small_talk"):
            logger.info(f"Skipping judge/repair for simple route: {route}")
            final = await self._naturalize(raw.strip())
            return final, {
                "route": route,
                "route_score": route_score,
                "repaired": False,
                "fast_path": True,
                "naturalized": True,
            }

        # 其餘路線 → Judge & Repair
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

    async def _handle_guided_flow(
        self, *, question: str, history: str, session_id: str = None
    ):
        """Handle guided flow for mh_support route"""
        logger.info("Handling guided flow for mh_support route")

        # 規劃（首次或尚未啟動 flow 時）
        plan_json = await self.flow.plan_if_needed(
            question=question, history=history, session_id=session_id
        )

        # RAG context（可選）
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

        # Hedge：main 與 fast 競速，取先回來的結果
        raw = await self._hedged_turn_generate(
            question=question,
            history=history,
            context=context,
            plan_json=plan_json,
            session_id=session_id,
        )

        # Judge & Repair（flow 專屬）
        spec = self.compiler.routes["mh_support"]
        constraints = self.compiler._join_constraints(spec.get("constraints", []))
        contract = self.compiler.contracts[spec["output_contract"]]
        tone_block = self.compiler.get_tone_block("balanced")
        banned = getattr(self.compiler, "banned_phrases", [])

        # 最近 E: 歷史（給 quick-check 用）
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

        qc_ok, qc_reason = self.judge.quick_check(
            assistant_raw=raw, banned_phrases=banned, recent_es=e_history
        )
        if not qc_ok:
            logger.info(f"Quick-check failed ({qc_reason}); attempting repair")
            try:
                fixed = await asyncio.wait_for(
                    self.repairer.repair(contract, constraints, question, raw), timeout=10.0
                )
                raw = fixed
            except Exception as e:
                logger.warning(f"Quick repair failed: {e}; using fast prompt")
                fast = await self._hedged_turn_generate(
                    question=question,
                    history=history,
                    context=context,
                    plan_json=plan_json,
                    session_id=session_id,
                )
                # 這個分支也自然化
                fast = await self._naturalize(fast.strip())
                return fast, {
                    "route": "mh_support",
                    "repaired": False,
                    "flow_active": True,
                    "quick_repair": True,
                    "naturalized": True,
                }

        # LLM judge（設時限避免卡住）
        try:
            ok = await asyncio.wait_for(
                self.judge.pass_fail(contract, constraints, question, raw), timeout=8.0
            )
        except asyncio.TimeoutError:
            logger.warning("Judge timed out; using fast repair path")
            ok = False

        if not ok:
            logger.info("Flow response failed judge, attempting repair")
            try:
                fixed = await asyncio.wait_for(
                    self.repairer.repair(contract, constraints, question, raw), timeout=10.0
                )
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
                fast = await self._naturalize(fast.strip())
                return fast, {
                    "route": "mh_support",
                    "repaired": False,
                    "flow_active": True,
                    "repair_error": str(e),
                    "fast_regen": True,
                    "naturalized": True,
                }

        # Tone scoring & optional tone repair
        try:
            tone_ok = await asyncio.wait_for(self.judge.tone_score(tone_block, raw), timeout=6.0) >= 0.7
        except asyncio.TimeoutError:
            tone_ok = True  # latency 保護：逾時就不修語氣

        if not tone_ok:
            try:
                raw = await asyncio.wait_for(
                    self.repairer.tone_repair(tone_block, contract, constraints, question, raw),
                    timeout=12.0,
                )
            except Exception as e:
                logger.warning(f"Tone repair failed: {e}")

        final = await self._naturalize(raw.strip())
        return final, {"route": "mh_support", "repaired": not ok, "flow_active": True, "naturalized": True}

    async def _hedged_turn_generate(
        self, *, question: str, history: str, context: str, plan_json: str, session_id: str = None
    ) -> str:
        """Race a main prompt vs a fast prompt; first result wins."""
        # 構建兩套提示
        state = self.flow.load_state(session_id) if self.flow else None
        technique = state.technique if state else ""
        step_index = state.step_index if state else 0
        expected_q = state.last_question_type if state else None

        # main turn
        main_prompt = self.compiler.compile_flow_turn(
            route="mh_support",
            question=question,
            history=history,
            context=context,
            technique=technique,
            step_index=step_index,
            plan_json=plan_json or "{}",
            expected_question_type=expected_q,
        )
        # fast turn（更短更嚴格）
        fast_prompt = self.compiler.compile_flow_turn_fast(
            route="mh_support",
            question=question,
            history=history,
            context=context,
            technique=technique,
            step_index=step_index,
            plan_json=plan_json or "{}",
            expected_question_type=expected_q,
        )

        async def _gen_main():
            return await self.main.complete(main_prompt, max_time=30.0, max_new_tokens=None)

        async def _gen_fast():
            return await self.main.complete(fast_prompt, max_time=8.0, max_new_tokens=80)

        # Hedge：先發 main，再延遲 2 秒發 fast；誰先回就用誰
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
            # 如果先完成的失敗，嘗試另一個
            for p in pending:
                try:
                    return await p
                except Exception:
                    pass
            raise
