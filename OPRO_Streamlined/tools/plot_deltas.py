import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib

# Non-interactive backend for CI/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def read_summary_csv(path: Path) -> Tuple[List[Tuple[int, float, float]], Dict[str, Any]]:
    rows: List[Tuple[int, float, float]] = []  # (iteration, avg, best)
    summary: Dict[str, Any] = {
        "final_score": None,
        "improvement_achieved": None,
        "total_iterations": None,
        "time_elapsed_s": None,
    }
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # comment footer lines, e.g. # summary_final_score,5.0
                parts = line[1:].split(",", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key == "summary_final_score":
                        summary["final_score"] = float(val) if val else None
                    elif key == "summary_improvement":
                        summary["improvement_achieved"] = float(val) if val else None
                    elif key == "total_iterations":
                        summary["total_iterations"] = int(float(val)) if val else None
                    elif key == "time_elapsed_s":
                        summary["time_elapsed_s"] = float(val) if val else None
                continue
            if line.startswith("iteration"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                it = int(float(parts[0]))
                avg = float(parts[1])
                best = float(parts[2])
                rows.append((it, avg, best))
    rows.sort(key=lambda x: x[0])
    return rows, summary


def safe_rel_impr(delta: float, base: float) -> float:
    if base is None or abs(base) < 1e-9:
        return 0.0
    return 100.0 * delta / base


def bar_with_delta(labels, values, title, outfile, annotate_extra: str = ""):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#7AA6DC", "#4C8EDA"])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel("Score")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if annotate_extra:
        plt.suptitle(annotate_extra, y=0.99, fontsize=9)
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_summary_progress_line(rows: List[Tuple[int, float, float]], baseline_sum: float, outfile: Path,
                               use_best: bool = True, title: str = "Summary-based progress") -> None:
    # Build x-axis labels and y values: Baseline + per-iteration series
    iterations = [r[0] for r in rows]
    per_iter_vals = [r[2] if use_best else r[1] for r in rows]  # best or avg
    x_labels = ["Baseline"] + [f"Iter {it}" for it in iterations]
    y_vals = [baseline_sum] + per_iter_vals

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(y_vals)), y_vals, marker="o", linewidth=2)
    plt.xticks(range(len(x_labels)), x_labels, rotation=0)
    plt.ylabel("Score")
    plt.title(title)
    # Annotate delta vs baseline
    delta = y_vals[-1] - y_vals[0]
    rel = 0.0 if abs(y_vals[0]) < 1e-9 else (100.0 * delta / y_vals[0])
    plt.suptitle(f"Δ={delta:.3f}  ({rel:.2f}%)", y=0.98, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outfile, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot delta charts from optimization_summary.csv")
    parser.add_argument(
        "--csv",
        type=str,
        default="OPRO_Streamlined/results/plots/optimization_summary.csv",
        help="Path to optimization_summary.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="OPRO_Streamlined/results/plots",
        help="Directory to save charts",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows, summary = read_summary_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    # Per-iteration best: first vs last
    first_it, first_avg, first_best = rows[0]
    last_it, last_avg, last_best = rows[-1]
    best_delta = last_best - first_best
    best_rel = safe_rel_impr(best_delta, first_best)

    # Summary-based baseline vs final
    final_score = summary.get("final_score")
    improvement = summary.get("improvement_achieved")
    if final_score is not None and improvement is not None:
        baseline_sum = final_score - improvement
        sum_delta = improvement
        sum_rel = safe_rel_impr(sum_delta, baseline_sum)
    else:
        # Fallback: use per-iteration average as baseline/final
        baseline_sum = first_avg
        final_score = last_avg
        sum_delta = final_score - baseline_sum
        sum_rel = safe_rel_impr(sum_delta, baseline_sum)

    # Plot 1: summary-based delta
    bar_with_delta(
        ["Baseline (summary)", "Final (summary)"],
        [baseline_sum, final_score],
        "Summary-based improvement",
        outdir / "summary_delta.png",
        annotate_extra=f"Δ={sum_delta:.3f}  ({sum_rel:.2f}%)",
    )

    # Plot 2: per-iteration best delta
    bar_with_delta(
        [f"Best@iter {first_it}", f"Best@iter {last_it}"],
        [first_best, last_best],
        "Per-iteration best: first vs last",
        outdir / "best_delta.png",
        annotate_extra=f"Δ={best_delta:.3f}  ({best_rel:.2f}%)",
    )

    # Plot 3: summary-based progress line (baseline -> per-iteration best)
    plot_summary_progress_line(
        rows,
        baseline_sum,
        outdir / "summary_progress.png",
        use_best=True,
        title="Summary-based progress (baseline → per-iteration best)",
    )

    print("Saved:")
    print(str(outdir / "summary_delta.png"))
    print(str(outdir / "best_delta.png"))
    print(str(outdir / "summary_progress.png"))


if __name__ == "__main__":
    main()


