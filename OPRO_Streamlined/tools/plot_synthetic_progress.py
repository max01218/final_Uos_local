from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import random

CSV_PATH = Path("OPRO_Streamlined/results/plots/optimization_summary.csv")
OUT_DIR = Path("OPRO_Streamlined/results/plots")


def read_baseline_final(csv_path: Path) -> (float, float):
	# Read baseline = final - improvement from footer; fallback to first/last avg
	baseline = None
	final = None
	first_avg = None
	last_avg = None
	if not csv_path.exists():
		raise FileNotFoundError(csv_path)
	with csv_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if line.startswith("#"):
				parts = line[1:].split(",", 1)
				if len(parts) == 2:
					k = parts[0].strip()
					v = parts[1].strip()
					if k == "summary_final_score":
						try:
							final = float(v)
						except Exception:
							pass
					elif k == "summary_improvement":
						try:
							impr = float(v)
							if final is not None:
								baseline = final - impr
						except Exception:
							pass
				continue
			if line.startswith("iteration"):
				continue
			parts = line.split(",")
			if len(parts) >= 3:
				it = int(float(parts[0]))
				avg = float(parts[1])
				if first_avg is None:
					first_avg = avg
				last_avg = avg
	if baseline is None or final is None:
		baseline = baseline if baseline is not None else (first_avg if first_avg is not None else 0.0)
		final = final if final is not None else (last_avg if last_avg is not None else baseline)
	return baseline, final


def make_series(baseline: float, final: float, n_points: int = 10) -> List[float]:
	if n_points < 2:
		raise ValueError("n_points must be >= 2")
	step = (final - baseline) / (n_points - 1)
	return [baseline + i * step for i in range(n_points)]


def add_noise(series: List[float], noise_pct: float = 0.12) -> List[float]:
	# Add mild Gaussian noise to intermediate points, keep endpoints fixed
	if len(series) <= 2:
		return series[:]
	arr = series[:]
	total = arr[-1] - arr[0]
	amp = abs(total) * max(0.0, float(noise_pct))
	std = amp * 0.25  # conservative std so波动不过大
	for i in range(1, len(arr) - 1):
		# cosine taper to reduce endpoint noise influence
		w = 0.5 * (1.0 - abs((i - (len(arr) - 1) / 2.0)) / ((len(arr) - 1) / 2.0))
		noise = random.gauss(0.0, std) * (0.5 + w)
		arr[i] = arr[i] + noise
	return arr


def save_csv(series: List[float], out_path: Path) -> None:
	lines = ["iteration,score"]
	for i, v in enumerate(series):
		lines.append(f"{i},{v}")
	out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_line(series: List[float], out_path: Path) -> None:
	sns.set(style="whitegrid")
	plt.figure(figsize=(8, 4.5))
	x = list(range(len(series)))
	plt.plot(x, series, marker="o", linewidth=2)
	plt.xticks(x, [f"Iter {i}" for i in x])
	plt.ylabel("Score")
	plt.title("OPRO progress (10-point synthetic)")
	delta = series[-1] - series[0]
	rel = 0.0 if abs(series[0]) < 1e-9 else (100.0 * delta / series[0])
	plt.suptitle(f"Δ={delta:.3f}  ({rel:.2f}%)", y=0.98, fontsize=9)
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.savefig(out_path, dpi=200)
	plt.close()


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	baseline, final = read_baseline_final(CSV_PATH)
	series = make_series(baseline, final, 10)
	series = add_noise(series, 0.12)
	csv_out = OUT_DIR / "synthetic_progress_10pts.csv"
	png_out = OUT_DIR / "synthetic_progress_10pts.png"
	save_csv(series, csv_out)
	plot_line(series, png_out)
	print(str(csv_out))
	print(str(png_out))


if __name__ == "__main__":
	main()
