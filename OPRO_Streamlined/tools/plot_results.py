import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_by_iteration(history: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for item in history:
        it = int(item.get("iteration", 0))
        grouped.setdefault(it, []).append(item)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def compute_iteration_stats(history: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[float]]:
    grouped = group_by_iteration(history)
    iterations: List[int] = []
    avg_scores: List[float] = []
    best_scores: List[float] = []
    for it, items in grouped.items():
        scores = [float(x.get("score", 0.0)) for x in items if x.get("score") is not None]
        if not scores:
            continue
        iterations.append(it)
        avg_scores.append(sum(scores) / len(scores))
        best_scores.append(max(scores))
    return iterations, avg_scores, best_scores


def plot_score_curves(iterations: List[int], avg_scores: List[float], best_scores: List[float], out_path: Path) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, avg_scores, marker="o", label="Average score")
    plt.plot(iterations, best_scores, marker="s", label="Best score")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("OPRO optimization scores over iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scores_hist(history: List[Dict[str, Any]], out_path: Path) -> None:
    scores = [float(x.get("score", 0.0)) for x in history if x.get("score") is not None]
    if not scores:
        return
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.histplot(scores, bins=20, kde=True)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Distribution of candidate scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_method_counts(history: List[Dict[str, Any]], out_path: Path) -> None:
    methods: Dict[str, int] = {}
    for x in history:
        m = (x.get("generation_method") or "unknown").strip()
        methods[m] = methods.get(m, 0) + 1
    if not methods:
        return
    names = list(methods.keys())
    counts = [methods[n] for n in names]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=names, y=counts)
    plt.xlabel("Generation method")
    plt.ylabel("Count")
    plt.title("Counts by generation method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_csv_summary(iterations: List[int], avg_scores: List[float], best_scores: List[float], summary: Dict[str, Any], out_path: Path) -> None:
    lines = ["iteration,avg_score,best_score"]
    for it, a, b in zip(iterations, avg_scores, best_scores):
        lines.append(f"{it},{a:.6f},{b:.6f}")
    # Append summary footer as comments
    lines.append("# summary_final_score," + str(summary.get("final_score", "")))
    lines.append("# summary_improvement," + str(summary.get("improvement_achieved", "")))
    lines.append("# total_iterations," + str(summary.get("total_iterations", "")))
    lines.append("# time_elapsed_s," + str(summary.get("time_elapsed", "")))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OPRO optimization results")
    parser.add_argument("--report", type=str, required=False, default="",
                        help="Path to optimization_report_*.json. If omitted, will auto-pick the latest in OPRO_Streamlined/results/")
    parser.add_argument("--outdir", type=str, required=False, default="OPRO_Streamlined/results/plots",
                        help="Directory to save generated plots")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
    else:
        # pick latest report in OPRO_Streamlined/results
        results_dir = repo_root / "OPRO_Streamlined" / "results"
        cand = sorted(results_dir.glob("optimization_report_*.json"))
        if not cand:
            raise FileNotFoundError(f"No optimization_report_*.json under {results_dir}")
        report_path = cand[-1]

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_report(report_path)
    history = list(data.get("optimization_history", []))
    summary = dict(data.get("summary", {}))

    iterations, avg_scores, best_scores = compute_iteration_stats(history)
    if not iterations:
        raise RuntimeError("No iteration statistics could be computed (empty history or missing scores)")

    # Plots
    plot_score_curves(iterations, avg_scores, best_scores, outdir / "score_over_iterations.png")
    plot_scores_hist(history, outdir / "scores_hist.png")
    plot_method_counts(history, outdir / "generation_method_counts.png")

    # CSV summary
    save_csv_summary(iterations, avg_scores, best_scores, summary, outdir / "optimization_summary.csv")

    print("Saved:")
    for p in [
        outdir / "score_over_iterations.png",
        outdir / "scores_hist.png",
        outdir / "generation_method_counts.png",
        outdir / "optimization_summary.csv",
    ]:
        print(str(p))


if __name__ == "__main__":
    main()


