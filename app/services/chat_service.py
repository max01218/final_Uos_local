import os
import time
import asyncio
import logging
from typing import Optional, Tuple
from fastapi import HTTPException
from app.schemas.chat import RAGRequest, RAGResponse
from app.services.intent_service import analyze_conversation_context
from app.services.memory_service import (
    ConversationStore,
    summarize_session_transcript,
)
from app.repositories.session_repo import (
    load_session_summary,
    save_session_summary,
    append_session_metrics,
)
from app.utils.postprocess import post_process_response
from app.utils.definitions import (
    is_definitional_question,
    load_definitional_prompt,
    get_medical_context_for_definition,
)
from app.utils.prompting import get_dynamic_prompt, get_step_by_step_prompt
from app.utils.crisis_detection import (
    detect_crisis_keywords,
    generate_crisis_response,
)
from app.services.rag_service import RAGService
from app.core.settings import settings

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, store, llm_client, conversation_store: ConversationStore, embedder=None):
        self.store = store
        self.llm = llm_client
        self.cs = conversation_store
        self.rag = RAGService(store, embedder=embedder)

    async def handle_chat(self, req: RAGRequest) -> Tuple[str, dict]:
        start_time = time.time()

        if detect_crisis_keywords(req.question):
            return generate_crisis_response(), {
                "status": "crisis_detected",
                "prompt_source": "crisis",
                "confidence": 1.0,
                "fusion_strategy": "crisis_intervention",
                "source_breakdown": {"crisis": 1.0},
                "follow_up_suggestions": ["Please call 999 immediately"],
                "safety_notes": ["User expressed suicidal thoughts - immediate intervention required"],
            }

        conversation_analysis = analyze_conversation_context(req.question, req.history)
        response_strategy = conversation_analysis["response_strategy"]

        session_id = req.session_id or str(int(time.time() * 1000))
        history = self.cs.get_conversation_history()
        session_summary = load_session_summary(session_id)

        # Pass strategy signals to RAG for weighting in future extension
        docs = self.rag.retrieve(req.question, k=5, signals={
            "response_strategy": response_strategy,
            "emotional_state": conversation_analysis.get("emotional_state"),
        })
        context = self.rag.build_context(docs, max_docs=2)
        if len(context) > 2000:
            context = context[:2000] + "..."

        # Detect step-by-step intent keywords (English/Chinese)
        step_keywords = [
            "step-by-step", "walk me through", "step by step", "steps",
            "Tonight", "Steps", "Teach me", "How to", "Show me how to"
        ]
        wants_steps = any(k in req.question.lower() for k in step_keywords)

        is_def = is_definitional_question(req.question)
        if is_def:
            prompt = load_definitional_prompt() or ""
            medical_context = get_medical_context_for_definition(req.question, self.store)
            if medical_context:
                context = medical_context
        else:
            # If user explicitly wants step-by-step, switch to step template
            if wants_steps:
                prompt = get_step_by_step_prompt()
            else:
                prompt = get_dynamic_prompt(req.type)

        combined_history = (session_summary + "\n\n" + history).strip() if session_summary else history
        limited_context = context[:1000] + "..." if len(context) > 1000 else context
        limited_history = combined_history[:500] + "..." if len(combined_history) > 500 else combined_history

        # Format prompt with variables and wrap to ChatML if needed
        try:
            formatted_prompt = prompt.format(
                context=limited_context,
                question=req.question,
                history=limited_history,
                response_strategy=response_strategy,
                tone=req.type,
            )
        except Exception:
            formatted_prompt = prompt

        if ("<|im_start|>" in formatted_prompt) or ("<|system|>" in formatted_prompt):
            pass
        else:
            formatted_prompt = (
                f"<|system|>\n{formatted_prompt}\n<|user|>\n{req.question}\n<|assistant|>\n"
            )

        try:
            answer = await self.llm.generate(formatted_prompt)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="LLM inference timed out")
        except Exception as e:
            logger.exception("LLM inference failed")
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {e}")
        answer = post_process_response(answer, req.question)

        # Post-trim: enforce single follow-up question and concise length, avoid mid-sentence cuts
        try:
            def _trim_to_complete_sentence(text: str, limit: int) -> str:
                if len(text) <= limit:
                    return text.strip()
                slice_text = text[:limit]
                # Prefer cutting at sentence boundary
                boundaries = ['.', '!', '?', '。', '！', '？']
                last_boundary = max(slice_text.rfind(b) for b in boundaries)
                if last_boundary >= 0:
                    return slice_text[: last_boundary + 1].strip()
                # Fallback: cut at last space
                last_space = slice_text.rfind(' ')
                if last_space >= 0:
                    return slice_text[: last_space].strip() + '…'
                return slice_text.strip() + '…'

            # Keep only the first Chinese question mark trail if multiple (legacy safety)
            if answer.count("？") > 1:
                first_q = answer.find("？")
                answer = answer[: first_q + 1]

            # Step-by-step responses: trim body smartly, preserve the final Q line if present
            if wants_steps:
                q_idx = answer.rfind('Q:')
                if q_idx != -1:
                    body = answer[:q_idx].strip()
                    qline = answer[q_idx:].strip()
                    body = _trim_to_complete_sentence(body, 240)
                    # Ensure Q line completeness and reasonable length
                    if len(qline) > 100:
                        qline = qline[:100].rstrip()
                        if not qline.endswith('?'):
                            qline = qline.rstrip('.') + '?'
                    elif not qline.endswith('?'):
                        qline = qline + '?'
                    answer = (body + '\n' + qline).strip()
                else:
                    answer = _trim_to_complete_sentence(answer, 240)
        except Exception:
            pass

        try:
            self.cs.add_interaction(
                user_message=req.question,
                assistant_message=answer,
                metadata={
                    'emotion': conversation_analysis.get("emotional_state"),
                    'response_strategy': response_strategy,
                    'confidence': None,
                    'fusion_strategy': None,
                    'weekly_goal': req.weekly_goal,
                    'feasibility': req.feasibility,
                    'anxiety_level': req.anxiety_level,
                },
                session_id=session_id,
            )
            if conversation_analysis.get("emotional_state"):
                self.cs.update_emotional_state(conversation_analysis["emotional_state"], 0.8, session_id=session_id)
        except Exception as e:
            logger.warning(f"memory update failed: {e}")

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

        meta = {
            "session_id": session_id,
            "intent": conversation_analysis.get("response_strategy"),
            "strategy": response_strategy,
            "tone_suggested": req.type,
            "weekly_goal": req.weekly_goal,
            "feasibility": req.feasibility,
            "anxiety_level": req.anxiety_level,
        }

        return answer, meta


