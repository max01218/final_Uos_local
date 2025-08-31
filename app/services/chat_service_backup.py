# app/services/chat_service.py
# All responses go through Qwen via prompts:
# 1) main therapist prompt -> 2) repair prompt (if needed) -> 3) minimal prompt (if needed).
# RAG + triage + tones + validator retained; rule-based fallback only on fatal errors.

import time
import logging
import asyncio
import re
from typing import Tuple, Optional
from fastapi import HTTPException
from textwrap import dedent
from app.utils.prompting import build_therapist_prompt, build_repair_prompt, build_minimal_esq_prompt, build_reflection_prompt
from app.utils.esq import format_esq, fallback_esq, humanize_esq
from app.schemas.chat import RAGRequest
from app.services.intent_service import analyze_conversation_context
from app.services.memory_service import ConversationStore, summarize_session_transcript
from app.repositories.session_repo import load_session_summary, save_session_summary
try:
    from app.repositories.session_repo import append_session_metrics  # type: ignore
except Exception:  # pragma: no cover
    append_session_metrics = None  # type: ignore

from app.utils.postprocess import post_process_response
from app.utils.crisis_detection import detect_crisis_keywords, generate_crisis_response
from app.services.rag_service import RAGService
from app.core.settings import settings


from app.utils.rag_triage import infer_topics, choose_technique, step_map_for
from app.utils.validators import validate_esq

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, store, llm_client, conversation_store: ConversationStore, embedder=None):
        self.store = store
        self.llm = llm_client
        self.cs = conversation_store
        try:
            self.rag = RAGService(store, embedder=embedder)
        except Exception as e:
            logger.warning(f"Failed to initialize RAGService: {e}, using fallback")
            self.rag = None

    async def handle_chat(self, req: RAGRequest) -> Tuple[str, dict]:
        t_total0 = time.time()

        # Crisis override
        if detect_crisis_keywords(req.question):
            return generate_crisis_response(), {
                "status": "crisis_detected",
                "prompt_source": "crisis",
                "confidence": 1.0,
                "fusion_strategy": "crisis_intervention",
                "source_breakdown": {"crisis": 1.0},
            }

        conversation_analysis = analyze_conversation_context(req.question, req.history)
        response_strategy = conversation_analysis["response_strategy"]

        session_id = req.session_id or str(int(time.time() * 1000))
        history_txt = self.cs.get_conversation_history()
        session_summary = load_session_summary(session_id)

        # --- Reflective Listening Turn ---
        try:
            interactions = self.cs.session_data.get(session_id, {})
            # The history is a list of (user, assistant) tuples
            turn_number = len(interactions.get("history", [])) if isinstance(interactions, dict) else 0
        except Exception:
            turn_number = 0

        if turn_number > 0 and settings.summary_every_n > 0 and turn_number % settings.summary_every_n == 0:
            logger.info(f"Turn {turn_number}: Triggering reflective listening response.")
            response_strategy = "reflect"
            
            prompt = build_reflection_prompt(req.question or "")
            prompt = f"<|system|>\n{prompt}\n<|user|>\n{req.question}\n<|assistant|>"

            raw_answer = ""
            llm_main_ms = 0
            try:
                if self.llm:
                    t_llm0_reflect = time.time()
                    raw_answer = await self.llm.generate(prompt, stop=["<END>"])
                    t_llm1_reflect = time.time()
                    llm_main_ms = int((t_llm1_reflect - t_llm0_reflect) * 1000)
            except Exception as e:
                logger.warning(f"LLM call for reflection failed: {e}")

            answer = format_esq(raw_answer)
            if not answer:
                answer = "E: It sounds like you're dealing with a lot right now.\nS: You mentioned feeling overwhelmed by the situation.\nQ: Did I get that right?"

            # Duplicate memory/meta logic and return
            self.cs.add_interaction(
                user_message=req.question or "",
                assistant_message=answer,
                metadata={"response_strategy": "reflect", "emotion": conversation_analysis.get("emotional_state")},
                session_id=session_id,
            )
            
            last_assistant = None
            try:
                if hasattr(self.cs, "get_last_assistant_message"):
                    last_assistant = self.cs.get_last_assistant_message(session_id)
            except Exception:
                last_assistant = None
            validator_metrics = validate_esq(answer, last_assistant=last_assistant)

            total_ms = int((time.time() - t_total0) * 1000)
            meta = {
                "session_id": session_id,
                "intent": response_strategy,
                "strategy": "reflect",
                "prompt_source": "reflection",
                "latency_ms": total_ms,
                "validator": validator_metrics,
                "perf": { "llm_main_ms": llm_main_ms },
            }
            return answer, meta

        # -------------------- RAG retrieval (timed) --------------------
        rag_retrieve_ms = rag_build_ms = 0
        doc_count = 0
        context = "No knowledge base available."
        docs = []
        t_rag0 = time.time()
        if self.rag:
            docs = self.rag.retrieve(
                req.question,
                k=5,
                signals={
                    "response_strategy": response_strategy,
                    "emotional_state": conversation_analysis.get("emotional_state"),
                },
            )
            t_retrieved = time.time()
            doc_count = len(docs) if docs else 0
            context = self.rag.build_context(docs, max_docs=2)
            t_built = time.time()
            rag_retrieve_ms = int((t_retrieved - t_rag0) * 1000)
            rag_build_ms = int((t_built - t_retrieved) * 1000)
        else:
            logger.warning("RAG service not available, using empty context")
        if len(context) > 2000:
            context = context[:2000] + "..."
        logger.info(f"[perf] RAG retrieve={rag_retrieve_ms}ms build_context={rag_build_ms}ms docs={doc_count} context_chars={len(context)}")

        # Continue detection: keyword or plain 0–10 rating
        lower_q = (req.question or "").lower()
        continue_flag = ("continue" in lower_q)
        if not continue_flag:
            m = re.search(r"\b(?:10|[0-9](?:\.\d)?)\b", lower_q)
            if m and self.cs.get_flag(session_id, "last_technique", None):
                continue_flag = True

        # Session flags
        last_technique: Optional[str] = self.cs.get_flag(session_id, "last_technique", None)
        technique_step_index: int = int(self.cs.get_flag(session_id, "technique_step_index", 0) or 0)

        # TRIAGE #1
        topics = infer_topics(docs)
        preferred_tech = choose_technique(
            topics, last_technique=last_technique, continue_flag=continue_flag
        )

        # Prompt assembly (therapist style)
        combined_history = (session_summary + "\n\n" + history_txt).strip() if session_summary else history_txt
        limited_context = context[:1000] + "..." if len(context) > 1000 else context
        limited_history = combined_history[:500] + "..." if len(combined_history) > 500 else combined_history

        # recent assistant openers as banned phrases (best-effort)
        recent_openers = []
        try:
            if hasattr(self.cs, "get_last_assistant_messages"):
                last_msgs = self.cs.get_last_assistant_messages(session_id, k=3)  # optional API
            else:
                last_msgs = []
            for m in last_msgs or []:
                first_line = (m.split("\n", 1)[0] or "").strip()
                if first_line:
                    recent_openers.append(first_line)
        except Exception:
            pass

        prompt = build_therapist_prompt(
            context=limited_context,
            question=req.question or "",
            history=limited_history,
            tone=req.type,
            topics=", ".join(topics[:3]) if topics else "",
            preferred_tech=preferred_tech or "",
            last_technique=last_technique or "",
            next_step_index=(technique_step_index + 1 if continue_flag and last_technique else 0),
            banned_phrases=recent_openers,
            fewshot="""
        Examples (do not copy; vary openings):

        [professional]
        E: Thanks for sharing—carrying that sounds exhausting.
        S: Try box breathing: inhale 4, hold 4, exhale 4, hold 4 — 4 cycles.
        Q: After that, where’s your tension from 0–10?

        [caring]
        E: I hear how overwhelming this feels right now.
        S: Place a hand on your chest and breathe slowly 60 seconds, noticing the rise and fall.
        Q: How does that feel in your body, 0–10?

        [balanced]
        E: Given what you said, it makes sense you’re tense tonight.
        S: Name 5 things you can see — 60 seconds.
        Q: Want to try the next step?
        """,
        )

        # Technique constraints for continue
        step_mode = "normal"
        if continue_flag and last_technique:
            prompt += (
                f"\n\n[STEP MAP]\n"
                "If TECHNIQUE CONTEXT is given, output ONLY the next micro-step for that technique.\n"
                f"{step_map_for(last_technique)}"
                "Do not restate previous steps. Keep exactly three lines per the contract."
            )
            step_mode = "continue"
        else:
            if last_technique:
                prompt += f"\n\n[UNSUITABLE METHODS]\n- {last_technique}\n"

        if ("<|im_start|>" not in prompt) and ("<|system|>" not in prompt):
            prompt = f"<|system|>\n{prompt}\n<|user|>\n{req.question}\n<|assistant|>\n"

        logger.info(
            f"[perf] Prompt built: prompt_chars={len(prompt)} "
            f"history_chars={len(limited_history)} context_chars={len(limited_context)} "
            f"topics={topics[:3] if topics else []} preferred_tech={preferred_tech} "
            f"last_technique={last_technique} step_mode={step_mode}"
        )

        # -------------------- LLM main generation (timed) --------------------
        llm_main_ms = llm_repair_ms = llm_minimal_ms = 0
        post_main_ms = post_repair_ms = post_minimal_ms = 0

        try:
            if self.llm:
                t_llm0 = time.time()
                raw = await self.llm.generate(prompt, stop=["<END>"])
                t_llm1 = time.time()
                llm_main_ms = int((t_llm1 - t_llm0) * 1000)
                logger.info(f"[perf] LLM main took {llm_main_ms}ms; raw_chars={len(raw)}")
            else:
                logger.warning("LLM not available, returning maintenance message")
                raw = "E: I'm sorry, the service is temporarily unavailable.\nS: Take one slow breath — inhale 4, exhale 6 — 1 cycle.\nQ: Can we try again in a moment?"
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="LLM inference timed out")
        except Exception as e:
            logger.exception("LLM inference failed")
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {e}")

        t_post0 = time.time()
        raw = post_process_response(raw, req.question or "")
        answer = format_esq(raw, word_limit=120)
        answer = humanize_esq(answer)
        t_post1 = time.time()
        post_main_ms = int((t_post1 - t_post0) * 1000)
        logger.info(f"[perf] Postprocess main took {post_main_ms}ms; esq_ok={bool(answer.strip())}")

        # -------------------- LLM repair generation (timed) --------------------
        repaired = ""
        if not answer.strip():
            repair_prompt = build_repair_prompt(raw=raw, question=req.question or "", tone=req.type or "balanced")
            try:
                t_llm2 = time.time()
                repaired = await self.llm.generate(repair_prompt, stop=["<END>"])
                t_llm3 = time.time()
                llm_repair_ms = int((t_llm3 - t_llm2) * 1000)
                logger.info(f"[perf] LLM repair took {llm_repair_ms}ms; repaired_chars={len(repaired)}")
            except Exception:
                repaired = ""

            t_post2 = time.time()
            answer = format_esq(repaired, word_limit=120)
            t_post3 = time.time()
            post_repair_ms = int((t_post3 - t_post2) * 1000)
            logger.info(f"[perf] Postprocess repair took {post_repair_ms}ms; esq_ok={bool(answer.strip())}")

        # -------------------- LLM minimal generation (timed) --------------------
        minimal_out = ""
        if not answer.strip():
            minimal_prompt = build_minimal_esq_prompt(req.question or "", tone=req.type or "balanced")
            try:
                t_llm4 = time.time()
                minimal_out = await self.llm.generate(minimal_prompt, stop=["<END>"])
                t_llm5 = time.time()
                llm_minimal_ms = int((t_llm5 - t_llm4) * 1000)
                logger.info(f"[perf] LLM minimal took {llm_minimal_ms}ms; minimal_chars={len(minimal_out)}")
            except Exception:
                minimal_out = ""

            t_post4 = time.time()
            answer = format_esq(minimal_out, word_limit=120) or minimal_out.strip()
            t_post5 = time.time()
            post_minimal_ms = int((t_post5 - t_post4) * 1000)
            logger.info(f"[perf] Postprocess minimal took {post_minimal_ms}ms; esq_len={len(answer)}")

        # Final fallback if all else fails
        if not answer.strip():
            logger.warning("All LLM generation attempts failed. Using smart fallback.")
            answer = fallback_esq(req.question or "")

        # Avoid Empathy Repetition
        try:
            e_line = (answer.splitlines()[0] if answer else "")
            if e_line.startswith("E:"):
                last_e = self.cs.get_flag(session_id, "last_empathy", "")
                # Check if the core message is the same, ignoring the "E: " prefix
                current_e_core = e_line[2:].strip().lower()
                last_e_core = (last_e[2:] if last_e.startswith("E:") else last_e).strip().lower()

                if last_e_core and current_e_core == last_e_core:
                    logger.info("Repeated empathy detected. Humanizing.")
                    answer = humanize_esq(answer) 
                    e_line = (answer.splitlines()[0] if answer else "")

                if e_line.startswith("E:"):
                     self.cs.set_flag(session_id, "last_empathy", e_line)
        except Exception as e:
            logger.warning(f"Failed to process empathy repetition check: {e}")

        # Validator
        last_assistant = None
        try:
            if hasattr(self.cs, "get_last_assistant_message"):
                last_assistant = self.cs.get_last_assistant_message(session_id)
        except Exception:
            last_assistant = None
        validator_metrics = validate_esq(answer, last_assistant=last_assistant)

        # Memory + technique flags
        try:
            self.cs.add_interaction(
                user_message=req.question or "",
                assistant_message=answer,
                metadata={
                    "emotion": conversation_analysis.get("emotional_state"),
                    "response_strategy": response_strategy,
                    "weekly_goal": req.weekly_goal,
                    "feasibility": req.feasibility,
                    "anxiety_level": req.anxiety_level,
                },
                session_id=session_id,
            )
            try:
                a_low = answer.lower()
                if any(k in a_low for k in ["inhale", "exhale", "breath", "breathing", "4 cycles"]):
                    current_technique = "breathing"
                elif any(k in a_low for k in ["progressive muscle relaxation", "pmr", "tense", "release 10s"]):
                    current_technique = "pmr"
                elif any(k in a_low for k in ["grounding", "5 things you can see", "5-4-3-2-1"]):
                    current_technique = "grounding"
                elif any(k in a_low for k in ["20 min", "20-minute", "stimulus control", "leave bed"]):
                    current_technique = "stimulus_control"
                else:
                    current_technique = self.cs.get_flag(session_id, "last_technique", None)

                if current_technique:
                    prev = int(self.cs.get_flag(session_id, "technique_step_index", 0) or 0)
                    if current_technique == self.cs.get_flag(session_id, "last_technique", None):
                        self.cs.set_flag(session_id, "technique_step_index", prev + 1)
                    else:
                        self.cs.set_flag(session_id, "technique_step_index", 1)
                    self.cs.set_flag(session_id, "last_technique", current_technique)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"memory update failed: {e}")

        # Summarization
        try:
            N = settings.summary_every_n
            interactions = self.cs.session_data.get(session_id, {})
            if isinstance(interactions, dict) and len(interactions) % N == 0:
                transcript = self.cs.get_conversation_history()
                summary = summarize_session_transcript(transcript)
                if summary:
                    self.cs.update_session_summary(session_id, summary)
                    save_session_summary(session_id, summary)
        except Exception as e:
            logger.warning(f"summary update failed: {e}")

        total_ms = int((time.time() - t_total0) * 1000)
        logger.info(
            f"[perf] TOTAL={total_ms}ms | RAG(retr:{rag_retrieve_ms}ms,build:{rag_build_ms}ms) "
            f"| LLM(main:{llm_main_ms}ms,repair:{llm_repair_ms}ms,min:{llm_minimal_ms}ms) "
            f"| POST(main:{post_main_ms}ms,repair:{post_repair_ms}ms,min:{post_minimal_ms}ms)"
        )

        meta = {
            "session_id": session_id,
            "intent": response_strategy,
            "strategy": response_strategy,
            "tone_suggested": req.type,
            "prompt_source": "therapist+tone+repair",
            "latency_ms": total_ms,
            "validator": validator_metrics,
            "perf": {
                "rag_retrieve_ms": rag_retrieve_ms,
                "rag_build_ms": rag_build_ms,
                "doc_count": doc_count,
                "llm_main_ms": llm_main_ms,
                "llm_repair_ms": llm_repair_ms,
                "llm_minimal_ms": llm_minimal_ms,
                "post_main_ms": post_main_ms,
                "post_repair_ms": post_repair_ms,
                "post_minimal_ms": post_minimal_ms,
                "prompt_chars": len(prompt),
                "context_chars": len(limited_context),
                "history_chars": len(limited_history),
            },
        }
        return answer, meta
