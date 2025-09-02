import textwrap

class PromptCompiler:
    def __init__(self, registry_path: str):
        # 保留你原本載入 registry.yaml 的邏輯；若簡化可內建模板
        self.routes = {
            "mh_support": {
                "output_contract": "esq_three_lines",
                "constraints": [
                    "Produce exactly three lines prefixed by E:, S:, Q:",
                    "E: one empathetic sentence only",
                    "S: one low-burden, concrete action (<=1 sentence)",
                    "Q: exactly one short question",
                ],
            },
            "info_definition": {
                "output_contract": "plain_text",
                "constraints": [
                    "2–4 sentences",
                    "No lists, no labels, no role prefixes",
                ],
            },
            "greeting": {"output_contract": "plain_text", "constraints": []},
            "other": {"output_contract": "plain_text", "constraints": []},
        }
        self.contracts = {
            "esq_three_lines": "E:\\nS:\\nQ:",
            "plain_text": "",
        }

    def _join_constraints(self, items):
        return "\n".join(items or [])

    def compile(self, *, route: str, question: str, history: str, context: str, tone: str):
        if route == "info_definition":
            # 最終仍會走 orchestrator 的 _gen_info_definition；此處保底
            return textwrap.dedent(f"""
            Provide a concise, plain-English explanation.
            - 2 to 4 sentences. No lists, no labels, no role prefixes.
            Question: {question}
            Answer:
            """).strip()
        elif route == "other":
            return textwrap.dedent(f"""
            Write a brief, helpful reply in plain English.
            - Keep under 120 words.
            Question: {question}
            Reply:
            """).strip()
        else:
            # mh_support/greeting 基本不從這裡出；保底
            return f"User: {question}\nAnswer briefly:"

    # Flow prompts（供 GuidedFlowService 使用）
    def compile_flow_plan(self, route: str, question: str, history: str):
        return textwrap.dedent(f"""
        You are a mental-health micro-coach. Create a short JSON plan (3–5 steps) for a gentle intervention.
        Include fields: technique, steps (array of strings).
        Keep under 100 words.

        User: {question}
        JSON:
        """).strip()

    def compile_flow_turn(self, **kw):
        return textwrap.dedent(f"""
        You are guiding one small step. Output exactly three lines with labels:
        E: one empathetic sentence.
        S: one concrete, low-burden action (one sentence).
        Q: exactly one short question.

        Keep it tight and specific. No extra lines.

        Context: {kw.get('context','')}
        Technique: {kw.get('technique','')}
        StepIndex: {kw.get('step_index',0)}
        PlanJSON: {kw.get('plan_json','')}
        User: {kw.get('question','')}
        """).strip()

    def compile_flow_turn_fast(self, **kw):
        return textwrap.dedent(f"""
        Output exactly:
        E: <one concise empathetic sentence>
        S: <one concrete, low-burden action in one sentence>
        Q: <one short question>

        User: {kw.get('question','')}
        """).strip()

    def compile_flow_wrap_up(self, **kw):
        return textwrap.dedent(f"""
        Summarize briefly what we tried and one next small step. 2 sentences. No labels.
        User: {kw.get('question','')}
        """).strip()

    # 供 flow_service 擷取
    def extract_technique(self, plan_json: str) -> str:
        import json
        try:
            data = json.loads(plan_json)
            return (data.get("technique") or "").strip()
        except Exception:
            return ""

    def extract_total_steps(self, plan_json: str) -> int:
        import json
        try:
            data = json.loads(plan_json)
            steps = data.get("steps") or []
            return max(0, int(len(steps)))
        except Exception:
            return 0
