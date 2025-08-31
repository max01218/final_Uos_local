# app/schemas/route.py
from pydantic import BaseModel, Field
from typing import List, Literal

RouteType = Literal[
    "greeting",        # salutations
    "small_talk",      # acknowledgements, short chit-chat
    "crisis",          # safety-critical
    "mh_support",      # mental-health support
    "info_definition", # definitional / factual
    "other"            # fallback
]

class RouteDecision(BaseModel):
    route: RouteType
    confidence: float = Field(ge=0.0, le=1.0)
    triggers: List[str] = []
