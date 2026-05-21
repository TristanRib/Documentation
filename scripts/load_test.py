import pandas as pd
import time
import tracemalloc
import os
from scripts.predict import predict_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transformed_data.csv")


df = pd.read_csv(DATA_PATH)
percentages = [20, 40, 60, 80, 100, 200, 300, 400, 500]

print(f"{'Samples':>16} {'Time':>10} {'Max RAM':>10} {'Anomalies':>12}")
print("-" * 58)

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

    size = len(df_test)
    n_anomalies = (result["iso_label"] == "anomalie").sum()
    samples_col = f"{size} ({pct}%)"
    print(
        f"{samples_col:>16} {total_time:>9.4f}s {peak_ram / 1024**2:>9.2f}MB "
        f"{n_anomalies:>6} ({100 * n_anomalies / size:.1f}%)"
    )
