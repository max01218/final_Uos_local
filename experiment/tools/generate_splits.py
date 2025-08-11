import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PLAN_PATH = REPO_ROOT / "experiment" / "config" / "split_plan.json"
PROMPTS_DIR = REPO_ROOT / "prompts"
ICD11_RAW_DIR = REPO_ROOT / "icd11_ch6_data" / "raw"
OUTPUT_DIR = REPO_ROOT / "experiment" / "runs" / "splits"


def read_split_plan(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_hash_float(key: str, seed: int) -> float:
    base = f"{seed}|{key}".encode("utf-8", errors="ignore")
    h = hashlib.sha256(base).hexdigest()
    as_int = int(h, 16)
    return as_int / float(2 ** 256)


def detect_language(text: str) -> str:
    # Very simple heuristic: presence of CJK → zh; else en
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


CRISIS_KEYWORDS_ZH = [
    "自殺", "自伤", "自殘", "自殘", "自殺念頭", "輕生", "傷害自己", "殺", "死亡", "不想活了", "結束生命",
    "危機", "緊急", "立即危險", "自殘", "自我傷害",
]
CRISIS_KEYWORDS_EN = [
    "suicide", "self-harm", "self harm", "kill myself", "kill others", "hurt myself", "end my life",
    "die", "death", "immediate danger", "emergency", "harm myself", "harm others",
]


def is_crisis_text(text: str, lang_hint: str = None) -> bool:
    text_lower = text.lower()
    if lang_hint == "zh" or (lang_hint is None and detect_language(text) == "zh"):
        return any(kw in text for kw in CRISIS_KEYWORDS_ZH)
    return any(kw in text_lower for kw in CRISIS_KEYWORDS_EN)


def safe_read_text(path: Path, limit_chars: int = 5000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
        return data[:limit_chars]
    except Exception:
        return ""


def extract_text_from_json(path: Path, limit_chars: int = 8000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
    except Exception:
        return ""

    parts: List[str] = []
    def collect(o: Any):
        if isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
        elif isinstance(o, str):
            parts.append(o)

    collect(obj)
    text = "\n".join(parts)
    return text[:limit_chars]


def load_samples(split_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    allowed_langs = set(split_plan.get("filters", {}).get("language", ["zh", "en"]))
    max_len = int(split_plan.get("filters", {}).get("max_length_chars", 2000))

    samples: List[Dict[str, Any]] = []

    # From prompts/
    if PROMPTS_DIR.exists():
        for fp in PROMPTS_DIR.rglob("*.txt"):
            rel = fp.relative_to(REPO_ROOT).as_posix()
            text = safe_read_text(fp, limit_chars=max_len + 1000)
            lang = detect_language(text)
            if lang not in allowed_langs:
                continue
            text_trim = text[:max_len]
            crisis = is_crisis_text(text_trim, lang)
            samples.append({
                "id": rel,
                "source_type": "user_prompts",
                "language": lang,
                "risk": "crisis" if crisis else "normal",
                "topic": "crisis" if crisis else "general",
            })

    # From icd11_ch6_data/raw/
    if ICD11_RAW_DIR.exists():
        for fp in ICD11_RAW_DIR.rglob("*.json"):
            rel = fp.relative_to(REPO_ROOT).as_posix()
            text = extract_text_from_json(fp, limit_chars=max_len + 2000)
            lang = detect_language(text)
            if lang not in allowed_langs:
                continue
            text_trim = text[:max_len]
            crisis = is_crisis_text(text_trim, lang)
            samples.append({
                "id": rel,
                "source_type": "icd11_raw",
                "language": lang,
                "risk": "crisis" if crisis else "normal",
                "topic": "crisis" if crisis else "general",
            })

    return samples


def assign_splits(samples: List[Dict[str, Any]], split_plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    seed = int(split_plan.get("seed", 0))
    splits = split_plan.get("splits", {"train": 0.6, "val": 0.2, "test": 0.2})
    train_r = float(splits.get("train", 0.6))
    val_r = float(splits.get("val", 0.2))
    test_r = float(splits.get("test", 0.2))
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Stratify by (source_type, risk)
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for s in samples:
        key = (s["source_type"], s["risk"])
        by_group.setdefault(key, []).append(s)

    out = {"train": [], "val": [], "test": []}
    for group_key, group_samples in by_group.items():
        for s in group_samples:
            key = f"{s['id']}|{s['source_type']}|{s['risk']}"
            r = compute_hash_float(key, seed)
            if r < train_r:
                out["train"].append(s)
            elif r < train_r + val_r:
                out["val"].append(s)
            else:
                out["test"].append(s)
    return out


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    def counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        total = len(rows)
        crisis = sum(1 for r in rows if r["risk"] == "crisis")
        icd = sum(1 for r in rows if r["source_type"] == "icd11_raw")
        prompts = sum(1 for r in rows if r["source_type"] == "user_prompts")
        return {
            "total": total,
            "crisis": crisis,
            "crisis_ratio": (crisis / total) if total else 0.0,
            "icd11_raw": icd,
            "user_prompts": prompts,
        }

    return {split: counts(rows) for split, rows in splits.items()}


def main() -> None:
    plan = read_split_plan(SPLIT_PLAN_PATH)
    samples = load_samples(plan)
    splits = assign_splits(samples, plan)
    ensure_output_dir(OUTPUT_DIR)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_jsonl(OUTPUT_DIR / "train.jsonl", splits["train"])
    write_jsonl(OUTPUT_DIR / "val.jsonl", splits["val"])
    write_jsonl(OUTPUT_DIR / "test.jsonl", splits["test"])

    summary = {
        "created_at": ts,
        "root": str(REPO_ROOT),
        "plan_path": str(SPLIT_PLAN_PATH.relative_to(REPO_ROOT)),
        "summary": summarize(splits),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


