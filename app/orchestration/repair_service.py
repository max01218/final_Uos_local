# app/orchestration/repair_service.py
from app.clients.llm_client import LLMClient

REPAIR_PROMPT = """Rewrite the assistant reply to satisfy the Output Contract and Constraints.
Keep the meaning; fix structure and style. Return only the final text.

Contract:
{contract}

Constraints:
{constraints}

User:
{question}

Assistant (raw):
{assistant_raw}

Repaired Reply:"""

TONE_REPAIR_PROMPT = """Rewrite the assistant reply to match the Tone Profile while preserving meaning and the Output Contract.
Return only the final text.

Tone Profile:
{tone_profile}

Contract:
{contract}

Constraints:
{constraints}

User:
{question}

Assistant (raw):
{assistant_raw}
"""

class RepairService:
    def __init__(self, client: LLMClient): 
        self.client = client
        
    async def repair(self, contract: str, constraints: str, question: str, assistant_raw: str) -> str:
        return await self.client.complete(REPAIR_PROMPT.format(
            contract=contract, 
            constraints=constraints, 
            question=question, 
            assistant_raw=assistant_raw
        ))

    async def tone_repair(self, tone_profile: str, contract: str, constraints: str, question: str, assistant_raw: str) -> str:
        return await self.client.complete(TONE_REPAIR_PROMPT.format(
            tone_profile=tone_profile,
            contract=contract,
            constraints=constraints,
            question=question,
            assistant_raw=assistant_raw
        ), max_time=12.0, max_new_tokens=160)
