# app/orchestration/prompt_compiler.py
import yaml
import json
from pathlib import Path
from textwrap import dedent

class PromptCompiler:
    def __init__(self, registry_path: str):
        self.data = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
        self.routes = self.data["routes"]
        self.fragments = self.data["fragments"]
        self.contracts = self.data["contracts"]
        self.banned_phrases = self.data.get("banned_phrases", [])
        self.anti_cliche_rules = self.data.get("anti_cliche_rules", "")

    def _join_constraints(self, keys):
        lines = []
        for k in keys or []:
            group, name = k.split(".")
            lines.append(self.fragments[group][name])
        return "\n".join(f"- {l}" for l in lines)

    def get_tone_block(self, tone: str) -> str:
        tones = self.data.get("tones", {})
        if not isinstance(tones, dict):
            return ""
        return tones.get((tone or "balanced").lower(), tones.get("balanced", ""))

    def get_tone_lexicon(self, tone: str) -> str:
        lex = self.data.get("tone_lexicon", {})
        tone_key = (tone or "balanced").lower()
        cfg = lex.get(tone_key, {}) if isinstance(lex, dict) else {}
        prefers = cfg.get("prefer", []) if isinstance(cfg, dict) else []
        banneds = cfg.get("banned", []) if isinstance(cfg, dict) else []
        lines = []
        if prefers:
            lines.append("Prefer terms: " + ", ".join(prefers))
        if banneds:
            lines.append("Avoid terms: " + ", ".join(banneds))
        return "\n".join(lines)

    def compile(self, *, route: str, question: str, history: str = "", context: str = "", tone: str = "balanced") -> str:
        spec = self.routes[route]
        system = spec["system"].strip()
        constraints = self._join_constraints(spec.get("constraints", []))
        contract = self.contracts[spec["output_contract"]].strip()
        tone_block = self.get_tone_block(tone)
        tone_lexicon = self.get_tone_lexicon(tone)

        # For simple routes, use minimal prompt with tone
        if route in ("greeting", "small_talk"):
            return f"System: {system}\n\nTone Profile:\n{tone_block}\n\nTone Lexicon:\n{tone_lexicon}\n\nConstraints:\n- Keep it under 10 words.\n- No emojis.\n- Avoid: {', '.join(self.banned_phrases)}\n\nUser: {question}\nAssistant:"
        
        # For complex routes, use full prompt with tone
        return dedent(f"""
        System: {system}
        
        Tone Profile:
        {tone_block}
        
        Tone Lexicon:
        {tone_lexicon}
        
        Anti-Cliche Rules:
        {self.anti_cliche_rules}
        Constraints: {constraints}
        - Avoid slogans and generic cheerleading.
        - One-question rule: exactly one question in Q line.
        Output: {contract}
        Context: {context}
        History: {history}
        
        User: {question}
        Assistant:""").strip()

    def _read_flow_template(self, rel_path: str) -> str:
        """Read flow template file"""
        try:
            return Path(rel_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Template not found: {rel_path}"

    def compile_flow_plan(self, *, route: str, question: str, history: str) -> str:
        """Compile planning prompt for guided flow"""
        template = self._read_flow_template("app/prompts/flows/mh_support_plan_prompt.md")
        return template.format(question=question, history=history)

    def compile_flow_turn(self, *, route: str, question: str, history: str, context: str,
                          technique: str, step_index: int, plan_json: str, expected_question_type: str, tone: str = "balanced") -> str:
        """Compile turn guidance prompt for guided flow"""
        template = self._read_flow_template("app/prompts/flows/mh_support_turn_prompt.md")
        tone_block = self.get_tone_block(tone)
        tone_lexicon = self.get_tone_lexicon(tone)
        return template.format(
            question=question, 
            history=history, 
            context=context,
            technique=technique or "", 
            step_index=step_index, 
            plan_json=plan_json or "{}",
            expected_question_type=expected_question_type,
            tone_profile=tone_block,
            tone_lexicon=tone_lexicon
        )

    def compile_flow_adjust(self, *, question: str, history: str, technique: str, problem: str) -> str:
        """Compile adjustment prompt when user is stuck"""
        template = self._read_flow_template("app/prompts/flows/mh_support_adjust_prompt.md")
        return template.format(
            question=question, 
            history=history, 
            technique=technique or "", 
            problem=problem
        )

    def compile_flow_wrap_up(self, *, question: str, history: str, technique: str) -> str:
        """Compile wrap-up prompt for flow completion"""
        template = self._read_flow_template("app/prompts/flows/mh_support_wrap_up_prompt.md")
        return template.format(
            question=question,
            history=history,
            technique=technique or ""
        )

    def compile_flow_turn_fast(self, *, route: str, question: str, history: str, context: str,
                               technique: str, step_index: int, plan_json: str, expected_question_type: str) -> str:
        """Compile a faster, stricter version of the turn prompt."""
        # Inline minimal strict template to avoid extra file
        template = (
            "Return EXACTLY three lines labeled E:, S:, Q:. STOP after Q line.\n"
            "Rules:\n"
            "- Max 12 words per line.\n"
            "- S must include timing (seconds/minutes) or repetitions.\n"
            "- Q must be one question using type: {expected_question_type}.\n"
            "Context: {context}\n"
            "Technique: {technique}\n"
            "Plan JSON: {plan_json}\n"
            "Step index: {step_index}\n"
            "User: {question}\n"
            "History: {history}\n"
        )
        return template.format(
            question=question, history=history, context=context,
            technique=technique or "", step_index=step_index,
            plan_json=plan_json or "{}", expected_question_type=expected_question_type
        )

    def extract_technique(self, plan_json: str) -> str:
        """Extract technique from plan JSON"""
        try:
            data = json.loads(plan_json)
            return data.get("technique", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    def extract_total_steps(self, plan_json: str) -> int:
        """Extract total steps count from plan JSON"""
        try:
            data = json.loads(plan_json)
            steps = data.get("steps", [])
            return len(steps) if isinstance(steps, list) else 3
        except (json.JSONDecodeError, TypeError):
            return 3
