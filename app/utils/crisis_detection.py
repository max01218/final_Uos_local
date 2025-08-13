from __future__ import annotations

from typing import List


CRISIS_KEYWORDS: List[str] = [
    "suicide",
    "kill myself",
    "hurt myself",
    "end it all",
    "want to die",
    "not worth living",
    "can't go on",
    "self harm",
    "self-harm",
]


def detect_crisis_keywords(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(keyword in lower for keyword in CRISIS_KEYWORDS)


def generate_crisis_response() -> str:
    return (
        "I care about your safety. Please contact your local emergency services or crisis helpline immediately."
        "In the UK, you can call the Samaritans free of charge at 116 123 (24/7) or text 'SHOUT' to 85258 for confidential support."
        "If you are in another country or region, please reach out to your local emergency services or suicide prevention helpline (such as the 988 Lifeline in the US)."
        "You are not alone — seeking help is a brave and important first step."
    )


