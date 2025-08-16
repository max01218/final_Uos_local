import os
import json
import time
import random
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "experiment" / "config" / "experiment_plan.yaml"
OUTPUT_RUNS_DIR = REPO_ROOT / "experiment" / "runs"
SPLITS_DIR = OUTPUT_RUNS_DIR / "splits"
RESULTS_DIR = OUTPUT_RUNS_DIR / "results"


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # lazy import
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_splits() -> Dict[str, List[Dict[str, Any]]]:
    def read_jsonl(p: Path) -> List[Dict[str, Any]]:
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    return {
        "train": read_jsonl(SPLITS_DIR / "train.jsonl"),
        "val": read_jsonl(SPLITS_DIR / "val.jsonl"),
        "test": read_jsonl(SPLITS_DIR / "test.jsonl"),
    }


def build_prompt(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def hash_id() -> str:
    return uuid.uuid4().hex


def request_openai_compatible(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, timeout_s: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {"content": content, "usage": usage, "latency_ms": latency_ms}


def infer_one(provider: Dict[str, Any], model_name: str, system_prompt: str, user_text: str, gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
    p_type = provider.get("type", "dry_run")
    if p_type == "openai_compatible":
        base_url = provider.get("base_url", "")
        if not base_url:
            raise RuntimeError("provider.base_url is empty. You can set env LLM_BASE_URL or add provider in YAML.")
        api_key = os.environ.get(provider.get("api_key_env", "OPENAI_API_KEY"), "")
        if not api_key:
            raise RuntimeError("API key not found in environment. Set env var (e.g., OPENAI_API_KEY) or configure provider.api_key_env.")
        timeout_s = int(provider.get("request_timeout_s", 60))
        return request_openai_compatible(base_url, api_key, model_name, build_prompt(system_prompt, user_text), gen_cfg["max_tokens"], gen_cfg["temperature"], gen_cfg["top_p"], timeout_s)
    elif p_type == "ollama":
        # Ollama native CLI (prefer generate, then fallback)
        import subprocess
        import json

        t0 = time.perf_counter()
        timeout_s = int(provider.get("request_timeout_s", 60))

        ollama_prompt = f"{system_prompt}\n\nUser: {user_text}\n\nAssistant:"

        def run_cmd(cmd_list):
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                encoding="utf-8",
            )
            result.check_returncode()
            return result.stdout

        # 方案A：generate -m -p --options
        cmd = [
            "ollama", "generate", "-m", model_name, "-p", ollama_prompt,
        ]
        if not provider.get("disable_options", False):
            cmd += [
                "--options",
                json.dumps(
                    {
                        "temperature": gen_cfg["temperature"],
                        "top_p": gen_cfg["top_p"],
                        "num_predict": gen_cfg["max_tokens"],
                    }
                ),
            ]
        try:
            content = run_cmd(cmd).strip()
        except subprocess.CalledProcessError as e1:
            err1 = (e1.stderr or "").lower()
            # Plan B: generate -m -p (without --options)
            try:
                content = run_cmd(["ollama", "generate", "-m", model_name, "-p", ollama_prompt]).strip()
            except subprocess.CalledProcessError as e2:
                err2 = (e2.stderr or "").lower()
                # Plan C: run <model> <prompt> (older versions allow prompt as a parameter)
                try:
                    content = run_cmd(["ollama", "run", model_name, ollama_prompt]).strip()
                except subprocess.CalledProcessError as e3:
                    raise RuntimeError(f"Ollama command failed. A: {err1} | B: {err2} | C: {e3.stderr}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        approx_tokens = max(1, (len(system_prompt) + len(user_text) + len(content)) // 4)
        usage = {
            "prompt_tokens": approx_tokens // 2,
            "completion_tokens": approx_tokens // 2,
            "total_tokens": approx_tokens,
        }
        return {"content": content, "usage": usage, "latency_ms": latency_ms}
            
    elif p_type == "dry_run":
        # Dry-run without external service; produce placeholder output for pipeline testing
        t0 = time.perf_counter()
        time.sleep(0.01)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        approx_tokens = max(1, (len(system_prompt) + len(user_text)) // 4)
        content = "[DRY_RUN] Placeholder output; configure provider to get real model results."
        usage = {"prompt_tokens": approx_tokens, "completion_tokens": 0, "total_tokens": approx_tokens}
        return {"content": content, "usage": usage, "latency_ms": latency_ms}
    else:
        raise NotImplementedError(f"Provider type not supported: {p_type}")


def price_cost(usage: Dict[str, Any], pricing: Dict[str, float]) -> float:
    pt = float(usage.get("prompt_tokens", 0))
    ct = float(usage.get("completion_tokens", 0))
    return (pt / 1000.0) * float(pricing.get("prompt_usd", 0.0)) + (ct / 1000.0) * float(pricing.get("completion_usd", 0.0))


def main() -> None:
    plan = read_yaml(PLAN_PATH)
    prompts_cfg = plan["prompts"]
    inf_cfg = plan["inference"]
    # Allow env-based provider when YAML lacks config; otherwise default to dry_run
    provider_cfg = inf_cfg.get("provider")
    if not provider_cfg:
        provider_cfg = {
            "type": os.environ.get("LLM_PROVIDER_TYPE", "dry_run"),
            "base_url": os.environ.get("LLM_BASE_URL", ""),
            "api_key_env": os.environ.get("LLM_API_KEY_ENV", "OPENAI_API_KEY"),
            "request_timeout_s": int(os.environ.get("LLM_TIMEOUT_S", "60")),
        }
    pricing_cfg = inf_cfg.get("pricing_per_1k_tokens") or {"prompt_usd": 0.0, "completion_usd": 0.0}

    system_prompts = {
        "P1": read_text(REPO_ROOT / prompts_cfg["P1_manual"]["path"]),
        "P2": read_text(REPO_ROOT / prompts_cfg["P2_ai_optimized"]["path"]),
        "P3": read_text(REPO_ROOT / prompts_cfg["P3_hybrid"]["path"]),
    }

    splits = load_splits()
    ensure_dir(RESULTS_DIR)

    rand = random.Random(plan["datasets"].get("seed", 0))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_id
    ensure_dir(run_dir)

    # Save run config snapshot
    (run_dir / "plan_snapshot.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # Choose which split to run; default test
    split_name = os.environ.get("EXPERIMENT_SPLIT", "test")
    dataset = splits[split_name]

    reps = int(inf_cfg.get("repetitions_per_sample", 1))
    temperature = float(inf_cfg.get("temperature", 0.2))
    top_p = float(inf_cfg.get("top_p", 1.0))
    max_tokens = int(inf_cfg.get("max_tokens", 512))
    randomize_order = bool(inf_cfg.get("randomize_prompt_order", True))

    model_name = inf_cfg["model_name"]
    provider_type = provider_cfg.get("type", "openai_compatible")

    out_jsonl = run_dir / f"pred_{split_name}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fo:
        for row in dataset:
            input_id = row["id"]
            user_text = input_id  # Baseline: use file path as proxy input; replace with file content if needed
            fp = REPO_ROOT / input_id
            if fp.suffix.lower() == ".txt" and fp.exists():
                user_text = fp.read_text(encoding="utf-8", errors="ignore")[:1000]

            variants = ["P1", "P2", "P3"]
            if randomize_order:
                rand.shuffle(variants)

            for variant in variants:
                system_prompt = system_prompts[variant]
                for rep in range(reps):
                    seed = plan["datasets"].get("seed", 0) + rep
                    rand.seed(seed)
                    gen_cfg = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

                    try:
                        result = infer_one(provider_cfg, model_name, system_prompt, user_text, gen_cfg)
                        usage = result.get("usage", {})
                        cost = price_cost(usage, pricing_cfg)
                        o = {
                            "run_id": run_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "input_id": input_id,
                            "input_source": row.get("source_type", "unknown"),
                            "prompt_variant": variant,
                            "prompt_version_info": "frozen",
                            "model_name": model_name,
                            "parameters": {
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_tokens": max_tokens,
                                "seed": seed,
                            },
                            "input_text": user_text,
                            "output_text": result["content"],
                            "token_usage": {
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "total_tokens": usage.get("total_tokens"),
                            },
                            "cost": cost,
                            "latency_ms": result.get("latency_ms"),
                            "errors": [],
                            "safety_flags": [],
                        }
                    except Exception as e:
                        o = {
                            "run_id": run_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "input_id": input_id,
                            "input_source": row.get("source_type", "unknown"),
                            "prompt_variant": variant,
                            "prompt_version_info": "frozen",
                            "model_name": model_name,
                            "parameters": {
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_tokens": max_tokens,
                                "seed": seed,
                            },
                            "input_text": user_text,
                            "output_text": "",
                            "token_usage": {},
                            "cost": 0.0,
                            "latency_ms": None,
                            "errors": [str(e)],
                            "safety_flags": [],
                        }

                    fo.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_jsonl}")


if __name__ == "__main__":
    main()


