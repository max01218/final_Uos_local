# app/utils/prompt_builders.py
from textwrap import dedent
from typing import Optional

def build_greeting_ack(question: str) -> str:
    return dedent(f"""
System:
You are a concise assistant. Acknowledge the greeting briefly.

User:
{question}

A:
""").strip()

def build_smalltalk_ack(question: str) -> str:
    return dedent(f"""
System:
You are a concise assistant. Acknowledge briefly without starting therapeutic steps.

User:
{question}

A:
""").strip()

def build_crisis_prompt(question: str, region_hint: Optional[str] = "UK") -> str:
    return dedent(f"""
System:
You are a safety-first assistant. Provide immediate safety guidance and ask a single safety check question.

User:
{question}

A:
""").strip()

def build_definitional_prompt(context: str, question: str) -> str:
    return dedent(f"""
System:
You are a professional information assistant. Provide a concise, factual answer aligned with medical classification when relevant.

Context:
{context}

User:
{question}

A:
""").strip()

def build_therapist_prompt(context: str, history: str, question: str, tone: str) -> str:
    return dedent(f"""
System:
You are a therapist-style assistant. Return exactly three lines labeled E, S, Q.
Follow real-voice constraints: one question only, varied openers, explicit timing/reps.

Context:
{context}

History:
{history}

User:
{question}

A:
""").strip()
