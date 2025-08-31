You are planning a short, structured micro-intervention for a mental-health support conversation.
Return STRICT JSON only.

Rules:
- Choose one technique family suitable for the user's described anxiety.
- Produce 3-5 micro-steps. Each step must include explicit timing/repetitions.
- Keep steps low burden and realistic for a single chat session.
- For each step, define the expected question type: "rating_0_10" or "yes_no".

Schema:
{{
  "technique": "<string>",
  "steps": [
    {{"s": "<one micro-step text>", "q_type": "rating_0_10"}},
    ...
  ]
}}

Available techniques:
- grounding_visual: Focus on immediate environment, visual anchoring
- breathing_box: Structured breathing patterns with counts
- PMR_hand: Progressive muscle relaxation for hands/arms
- cognitive_reframe: Simple thought challenging
- mindful_pause: Brief mindfulness moments

User:
{question}

History:
{history}
