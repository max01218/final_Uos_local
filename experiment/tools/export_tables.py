import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "experiment" / "reports"
TABLES_DIR = REPORTS_DIR / "tables"


def latest_report(prefix: str) -> Path:
    paths = sorted(REPORTS_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No report found with prefix: {prefix}_*.json under {REPORTS_DIR}")
    return paths[-1]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_metrics_csv(metrics: Dict[str, Any], out_path: Path) -> None:
    variants = metrics.get("variants", {})
    order = ["P1", "P2", "P3"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Variant", "Count", "Errors", "AvgLatencyMs", "p50", "p90",
            "TaskSuccess", "Factual", "Safety", "Consistency",
        ])
        for v in order:
            if v not in variants:
                continue
            it = variants[v]
            w.writerow([
                v,
                it.get("count", 0),
                it.get("errors", 0),
                round(float(it.get("avg_latency_ms", 0.0)), 2),
                round(float(it.get("latency_p50_ms", 0.0)), 2),
                round(float(it.get("latency_p90_ms", 0.0)), 2),
                round(float(it.get("task_success", 0.0)), 3),
                round(float(it.get("factual_alignment", 0.0)), 3),
                round(float(it.get("safety_pass", 0.0)), 3),
                round(float(it.get("consistency", 0.0)), 3),
            ])


def export_pairwise_csv(judge: Dict[str, Any], out_path: Path) -> None:
    pw = judge.get("summary", {}).get("pairwise_winrate", {})
    order = ["P1_vs_P2", "P1_vs_P3", "P2_vs_P3"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Comparison", "Winrate_A_over_non_ties"])
        for k in order:
            if k in pw:
                w.writerow([k, pw.get(k)])


def export_metrics_tex(metrics: Dict[str, Any], out_path: Path) -> None:
    variants = metrics.get("variants", {})
    order = ["P1", "P2", "P3"]
    lines: List[str] = []
    lines.append("% Auto-generated metrics table")
    lines.append("\\begin{tabular}{lrrrrrrrrr}")
    lines.append("\\hline")
    header = (
        "Variant & Count & Errors & AvgLatency & p50 & p90 & "
        "TaskSucc & Factual & Safety & Consist \\\\"  
    )
    lines.append(header)
    lines.append("\\hline")
    for v in order:
        if v not in variants:
            continue
        it = variants[v]
        row = (
            f"{v} & {int(it.get('count', 0))} & {int(it.get('errors', 0))} & "
            f"{float(it.get('avg_latency_ms', 0.0)):.2f} & "
            f"{float(it.get('latency_p50_ms', 0.0)):.2f} & "
            f"{float(it.get('latency_p90_ms', 0.0)):.2f} & "
            f"{float(it.get('task_success', 0.0)):.3f} & "
            f"{float(it.get('factual_alignment', 0.0)):.3f} & "
            f"{float(it.get('safety_pass', 0.0)):.3f} & "
            f"{float(it.get('consistency', 0.0)):.3f} \\\\"  
        )
        lines.append(row)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.write_text("\n\n".join(lines), encoding="utf-8")


def export_pairwise_tex(judge: Dict[str, Any], out_path: Path) -> None:
    pw = judge.get("summary", {}).get("pairwise_winrate", {})
    order = ["P1_vs_P2", "P1_vs_P3", "P2_vs_P3"]
    lines: List[str] = []
    lines.append("% Auto-generated pairwise table")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\hline")
    lines.append("Comparison & Winrate(A over non-ties) \\\\"
    )
    lines.append("\\hline")
    for k in order:
        if k in pw:
            lines.append(f"{k} & {float(pw[k]):.4f} \\\\" )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.write_text("\n\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = latest_report("metrics")
    judge_path = latest_report("judge")
    metrics = load_json(metrics_path)
    judge = load_json(judge_path)

    export_metrics_csv(metrics, TABLES_DIR / "metrics_latest.csv")
    export_pairwise_csv(judge, TABLES_DIR / "pairwise_latest.csv")
    export_metrics_tex(metrics, TABLES_DIR / "metrics_latest.tex")
    export_pairwise_tex(judge, TABLES_DIR / "pairwise_latest.tex")

    print(str(TABLES_DIR / "metrics_latest.csv"))
    print(str(TABLES_DIR / "pairwise_latest.csv"))
    print(str(TABLES_DIR / "metrics_latest.tex"))
    print(str(TABLES_DIR / "pairwise_latest.tex"))


if __name__ == "__main__":
    main()


