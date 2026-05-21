import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
import os
from scripts.predict import predict_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transformed_data.csv")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "load_test.png")

df = pd.read_csv(DATA_PATH)
percentages = [20, 40, 60, 80, 100, 200, 300, 400, 500]

print(f"Dataset de base : {len(df)} observations\n")
print(f"{'Samples':>16} {'Time':>10} {'Max RAM':>10} {'Anomalies':>12}")
print("-" * 58)

peak_rams = []
times = []

for pct in percentages:
    multiplier, remainder = divmod(pct, 100)
    parts = [df] * multiplier
    if remainder:
        parts.append(df.iloc[: len(df) * remainder // 100])
    df_test = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].reset_index(drop=True)

    tracemalloc.start()
    start = time.time()

    result = predict_dataset(df_test)

    total_time = time.time() - start
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_ram / 1024**2
    peak_rams.append(peak_ram_mb)
    times.append(total_time)

    size = len(df_test)
    n_anomalies = (result["iso_label"] == "anomalie").sum()
    samples_col = f"{size} ({pct}%)"
    print(
        f"{samples_col:>16} {total_time:>9.4f}s {peak_ram_mb:>9.2f}MB "
        f"{n_anomalies:>6} ({100 * n_anomalies / size:.1f}%)"
    )

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(percentages, peak_rams, marker="o", linewidth=2, color="tab:blue", label="RAM (MB)")
for pct, ram in zip(percentages, peak_rams):
    ax1.annotate(f"{ram:.1f} MB", (pct, ram), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
ax1.set_xlabel("% du dataset")
ax1.set_ylabel("Pic de RAM (MB)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(percentages, times, marker="s", linewidth=2, color="tab:orange", label="Temps (s)")
for pct, t in zip(percentages, times):
    ax2.annotate(f"{t:.2f}s", (pct, t), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color="tab:orange")
ax2.set_ylabel("Temps de réponse (s)", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

ax1.axvline(x=100, color="tab:red", linestyle="--", linewidth=1, label="100% (dataset complet)")
ax1.set_xticks(percentages)
ax1.set_xticklabels([f"{p}%" for p in percentages])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("RAM et temps de réponse selon la charge")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"\nGraphique sauvegardé : {PLOT_PATH}")
