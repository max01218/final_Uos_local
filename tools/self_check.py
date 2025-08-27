# tools/self_check.py
import re, sys, importlib, inspect
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OK, NG = "✅","❌"
def imp(m): 
    try: return importlib.import_module(m), None
    except Exception as e: return None, f"{NG} import {m} failed: {e}"
def has(p,pat): return re.search(pat, Path(p).read_text(encoding="utf-8",errors="ignore"), re.M)

print("=== Qwen Prompt Pipeline Self-Check ===")
# settings
settings, err = imp("app.core.settings"); 
if err: sys.exit(err)
s = settings.settings
print(OK,"settings loaded:", s.llm_model_id)
print(OK if "qwen" in s.llm_model_id.lower() else NG, "Qwen model id")
print("decode:", s.llm_temperature, s.llm_top_p, s.llm_repetition_penalty, s.llm_max_new_tokens)

# bootstrap
bp = ROOT/"app/bootstrap.py"
print(OK if bp.exists() else NG, "bootstrap.py exists")
print(OK if has(bp, r"pipeline\(") and has(bp, r"temperature") and has(bp,r"top_p") and has(bp,r"repetition_penalty") else NG,
      "bootstrap sets decoding params")
print(OK if has(bp, r"attn_implementation|attn_impl") else NG, "attention impl (sdpa/flash_attention_2)")

# adapter
adapter, err = imp("app.clients.llm_adapter"); 
if err: sys.exit(err)
print(OK if hasattr(adapter,"LLMAdapter") else NG, "LLMAdapter present")
sig = inspect.signature(adapter.LLMAdapter.generate)
print(OK if "stop" in sig.parameters else NG, "LLMAdapter.generate(stop=...)")
print(OK if "apply_chat_template" in inspect.getsource(adapter.LLMAdapter) or "<|assistant|>" in inspect.getsource(adapter.LLMAdapter) else NG,
      "Qwen chat template / ChatML")

# prompting
promp, err = imp("app.utils.prompting"); 
if err: sys.exit(err)
for fn in ["build_therapist_prompt","build_repair_prompt","build_minimal_esq_prompt"]:
    print(OK if hasattr(promp, fn) else NG, f"prompting.{fn}")
pp = ROOT/"app/utils/prompting.py"
print(OK if has(pp, r"\[OUTPUT CONTRACT\]") and has(pp, r"\[EXAMPLES\]") else NG, "contract+examples")

# chat service
cs = ROOT/"app/services/chat_service.py"
print(OK if cs.exists() else NG, "chat_service.py exists")
txt = cs.read_text(encoding="utf-8",errors="ignore") if cs.exists() else ""
print(OK if "build_therapist_prompt" in txt else NG, "main prompt path")
print(OK if "build_repair_prompt" in txt else NG, "repair path")
print(OK if "build_minimal_esq_prompt" in txt else NG, "minimal path")
print(OK if re.search(r'llm\.generate\([^)]*stop=\s*\["<END>"\]\)', txt) else NG, 'stop=["<END>"] passed')
print(OK if all(k in txt for k in ["infer_topics","choose_technique"]) else NG, "RAG triage wired")
print(OK if "validate_esq(" in txt and "meta" in txt else NG, "validator in meta")

# esq + validators
esq, err = imp("app.utils.esq"); print(OK if not err else NG, "esq import")
val, err = imp("app.utils.validators"); print(OK if not err else NG, "validators import")
print("=== Done ===")
