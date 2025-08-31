# app/utils/prompt_registry.py
from typing import Callable, Dict
from app.utils.prompt_builders import (
    build_greeting_ack,
    build_smalltalk_ack,
    build_crisis_prompt,
    build_definitional_prompt,
    build_therapist_prompt,
)

PromptBuilder = Callable[..., str]

ROUTE_TO_BUILDER: Dict[str, PromptBuilder] = {
    "greeting": build_greeting_ack,
    "small_talk": build_smalltalk_ack,
    "crisis": build_crisis_prompt,
    "info_definition": build_definitional_prompt,
    "mh_support": build_therapist_prompt,
    "other": build_therapist_prompt,
}
