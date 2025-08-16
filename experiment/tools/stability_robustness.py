import os
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_RESULTS_DIR = REPO_ROOT / "experiment" / "runs" / "results"
REPORTS_DIR = REPO_ROOT / "experiment" / "reports"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def latest_run_pred_path(split_name: str = "test") -> Tuple[str, Path]:
    runs = sorted([p for p in RUNS_RESULTS_DIR.glob("*") if p.is_dir()], key=lambda p: p.name)
    if not runs:
        raise RuntimeError("No results found under experiment/runs/results")
    latest = runs[-1]
    pred = latest / f"pred_{split_name}.jsonl"
    if not pred.exists():
        raise RuntimeError(f"Not found: {pred}")
    return latest.name, pred


def jaccard_similarity(a: str, b: str) -> float:
    import re
    tok = lambda s: set(re.findall(r"\b\w{4,}\b", (s or "").lower()))
    aa, bb = tok(a), tok(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def group_by_input_variant(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    g: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[(r.get("input_id"), r.get("prompt_variant"))].append(r)
    return g


def compute_stability(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    g = group_by_input_variant(rows)
    per_variant: Dict[str, Dict[str, Any]] = {}

    # For each variant, for each input, compute pairwise similarity across repetitions
    sims_by_variant: Dict[str, List[float]] = defaultdict(list)
    lat_by_variant: Dict[str, List[float]] = defaultdict(list)

    for (input_id, variant), items in g.items():
        outputs = [it.get("output_text", "") for it in items]
        lats = [float(it.get("latency_ms") or 0.0) for it in items]
        # Pairwise similarities
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sims_by_variant[variant].append(jaccard_similarity(outputs[i], outputs[j]))
        lat_by_variant[variant].extend(lats)

    summary: Dict[str, Any] = {}
    for variant in sorted(set(v for _, v in g.keys())):
        sims = sims_by_variant.get(variant, [])
        lats = lat_by_variant.get(variant, [])
        summary[variant] = {
            "stability_mean": round(statistics.mean(sims), 3) if sims else None,
            "stability_median": round(statistics.median(sims), 3) if sims else None,
            "stability_count_pairs": len(sims),
            "latency_ms_mean": round(statistics.mean(lats), 2) if lats else None,
            "latency_ms_p50": round(statistics.median(lats), 2) if lats else None,
        }
    return summary


def compute_robustness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Slice by simple metadata if available; here we use input_source and message length bins
    bins = {
        "short": lambda s: len(s) < 200,
        "medium": lambda s: 200 <= len(s) < 600,
        "long": lambda s: len(s) >= 600,
    }
    by_slice: Dict[str, Dict[str, Any]] = {}

    for slice_name, pred in bins.items():
        filt = [r for r in rows if pred(r.get("input_text", ""))]
        if not filt:
            by_slice[slice_name] = {"count": 0}
            continue
        # Reuse stability on the slice
        by_slice[slice_name] = {"count": len(filt), "stability": compute_stability(filt)}
    return by_slice


def main() -> None:
    ensure_dir(REPORTS_DIR)
    run_id, pred_path = latest_run_pred_path(split_name=os.environ.get("EXPERIMENT_SPLIT", "test"))
    rows = read_jsonl(pred_path)

    stability = compute_stability(rows)
    robustness = compute_robustness(rows)

    out = {
        "run_dir": run_id,
        "stability": stability,
        "robustness": robustness,
    }
    (REPORTS_DIR / f"stability_robustness_{run_id}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved stability/robustness report to {REPORTS_DIR / f'stability_robustness_{run_id}.json'}")


if __name__ == "__main__":
    main()


