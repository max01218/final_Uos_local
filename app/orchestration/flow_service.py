# app/orchestration/flow_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import re
from app.services.memory_service import ConversationStore
from app.orchestration.prompt_compiler import PromptCompiler
from app.clients.llm_client import LLMClient

# Regex patterns for user feedback detection
DONE_RE = re.compile(r"\b(done|finished|completed|ok|okay|yes|continue)\b", re.I)
RATING_RE = re.compile(r"\b([0-9]|10)\b")
STUCK_RE = re.compile(r"\b(difficult|hard|stuck|cannot|can't|unable|too much)\b", re.I)

@dataclass
class FlowState:
    """State management for guided flow sessions"""
    active: bool = False
    technique: Optional[str] = None
    step_index: int = 0
    last_rating: Optional[int] = None
    last_question_type: str = "rating_0_10"
    plan_json: str = ""
    total_steps: int = 0

class GuidedFlowService:
    """Manages guided micro-step interventions for mental health support"""
    
    def __init__(self, cs: ConversationStore, compiler: PromptCompiler, llm: LLMClient):
        self.cs = cs
        self.compiler = compiler
        self.llm = llm

    def load_state(self, session_id: Optional[str] = None) -> FlowState:
        """Load current flow state from conversation store"""
        return FlowState(
            active=bool(self.cs.get_flag(session_id, "flow_active") or False),
            technique=self.cs.get_flag(session_id, "technique"),
            step_index=int(self.cs.get_flag(session_id, "step_index") or 0),
            last_rating=self.cs.get_flag(session_id, "last_rating"),
            last_question_type=self.cs.get_flag(session_id, "last_question_type") or "rating_0_10",
            plan_json=self.cs.get_flag(session_id, "plan_json") or "",
            total_steps=int(self.cs.get_flag(session_id, "total_steps") or 0),
        )

    def save_state(self, s: FlowState, session_id: Optional[str] = None):
        """Save flow state to conversation store"""
        self.cs.set_flag(session_id, "flow_active", s.active)
        self.cs.set_flag(session_id, "technique", s.technique)
        self.cs.set_flag(session_id, "step_index", s.step_index)
        self.cs.set_flag(session_id, "last_rating", s.last_rating)
        self.cs.set_flag(session_id, "last_question_type", s.last_question_type)
        self.cs.set_flag(session_id, "plan_json", s.plan_json)
        self.cs.set_flag(session_id, "total_steps", s.total_steps)

    def _extract_rating(self, text: str) -> Optional[int]:
        """Extract 0-10 rating from user text"""
        m = RATING_RE.search(text)
        if not m:
            return None
        val = int(m.group(1))
        return val if 0 <= val <= 10 else None

    def ingest_user_feedback(self, user_text: str) -> Dict[str, Any]:
        """Parse user feedback for continue signals, ratings, and difficulty indicators"""
        rating = self._extract_rating(user_text)
        cont = bool(DONE_RE.search(user_text))
        stuck = bool(STUCK_RE.search(user_text))
        return {
            "rating": rating,
            "continue": cont,
            "stuck": stuck
        }

    async def plan_if_needed(self, *, question: str, history: str, session_id: Optional[str] = None) -> Optional[str]:
        """Create initial plan if flow is not active yet"""
        s = self.load_state(session_id)
        if s.active and s.technique:
            return s.plan_json
        
        # First time: let LLM choose technique and create 3-5 step plan
        prompt = self.compiler.compile_flow_plan(
            route="mh_support", 
            question=question, 
            history=history
        )
        # Time-box the initial plan to keep session responsive
        plan_json = await self.llm.complete(prompt, max_time=20.0, max_new_tokens=200)
        
        # Extract and save technique and plan
        technique = self.compiler.extract_technique(plan_json)
        total_steps = self.compiler.extract_total_steps(plan_json)
        
        # Initialize flow state
        s.active = True
        s.technique = technique
        s.step_index = 0
        s.last_question_type = "rating_0_10"
        s.plan_json = plan_json
        s.total_steps = total_steps
        
        self.save_state(s, session_id)
        return plan_json

    async def next_turn(self, *, question: str, history: str, context: str, 
                       plan_json: Optional[str], session_id: Optional[str] = None) -> str:
        """Generate next guided turn response"""
        s = self.load_state(session_id)
        
        # Check if we need to wrap up
        if s.step_index >= s.total_steps:
            return await self._generate_wrap_up(question=question, history=history, 
                                              technique=s.technique, session_id=session_id)
        
        prompt = self.compiler.compile_flow_turn(
            route="mh_support",
            question=question,
            history=history,
            context=context,
            technique=s.technique or "",
            step_index=s.step_index,
            plan_json=plan_json or s.plan_json,
            expected_question_type=s.last_question_type,
        )
        
        response = await self.llm.complete(prompt)
        
        # Clean up response - remove common prefixes and extract only E:S:Q: lines
        response = self._clean_esq_response(response)
        
        return response

    async def _generate_wrap_up(self, *, question: str, history: str, 
                               technique: str, session_id: Optional[str] = None) -> str:
        """Generate wrap-up summary and reset flow state"""
        prompt = self.compiler.compile_flow_wrap_up(
            question=question,
            history=history,
            technique=technique
        )
        
        response = await self.llm.complete(prompt)
        
        # Clean up response
        response = self._clean_esq_response(response)
        
        # Reset flow state
        s = FlowState()  # Reset to defaults
        self.save_state(s, session_id)
        
        return response

    async def handle_adjustment(self, *, question: str, history: str, problem: str, 
                               session_id: Optional[str] = None) -> str:
        """Handle when user is stuck or struggling with current step"""
        s = self.load_state(session_id)
        
        prompt = self.compiler.compile_flow_adjust(
            question=question,
            history=history,
            technique=s.technique or "",
            problem=problem
        )
        
        response = await self.llm.complete(prompt)
        
        # Clean up response
        response = self._clean_esq_response(response)
        
        # Switch to yes/no question for next turn
        s.last_question_type = "yes_no"
        self.save_state(s, session_id)
        
        return response

    def advance_or_adjust(self, user_text: str, session_id: Optional[str] = None):
        """Update flow state based on user feedback"""
        s = self.load_state(session_id)
        fb = self.ingest_user_feedback(user_text)
        
        if fb["rating"] is not None:
            s.last_rating = fb["rating"]
            if s.last_rating <= 3:
                # Low rating: stay on current step or step back
                s.step_index = max(0, s.step_index)
                s.last_question_type = "yes_no"  # Ask if they want to try something easier
            else:
                # Good rating: advance to next step
                s.step_index += 1
                s.last_question_type = "rating_0_10"
        elif fb["continue"]:
            # Explicit continue signal
            s.step_index += 1
            s.last_question_type = "rating_0_10"
        elif fb["stuck"]:
            # User indicating difficulty
            s.last_question_type = "yes_no"
        else:
            # No clear signal: switch question type to get clarity
            s.last_question_type = "yes_no"
        
        self.save_state(s, session_id)

    def complete_if_finished(self, session_id: Optional[str] = None, max_steps: int = 5) -> bool:
        """Check if flow should be completed"""
        s = self.load_state(session_id)
        return s.step_index >= max(s.total_steps, max_steps)

    def is_flow_active(self, session_id: Optional[str] = None) -> bool:
        """Check if guided flow is currently active"""
        s = self.load_state(session_id)
        return s.active

    def _clean_esq_response(self, response: str) -> str:
        """Clean and extract only the E:S:Q: lines from LLM response"""
        response = response.strip()
        
        # Remove common prefixes
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()
        
        lines = response.split('\n')
        esq_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('E:') or line.startswith('S:') or line.startswith('Q:'):
                esq_lines.append(line)
                # Stop after we have all three lines
                if len(esq_lines) == 3:
                    break
        
        # If we found valid E:S:Q: lines, return them
        if len(esq_lines) >= 3:
            return '\n'.join(esq_lines[:3])
        
        # Fallback: return first 3 lines if no proper E:S:Q: format found
        return '\n'.join(lines[:3])
    
