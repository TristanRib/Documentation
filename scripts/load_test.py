import pandas as pd
import time
import tracemalloc
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import predict_dataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transformed_data.csv")

df = pd.read_csv(DATA_PATH)

for i in range(5):
    length = len(df)
    df_test = df.iloc[: length // 5 * (i + 1)]

    tracemalloc.start()
    start = time.time()

    result = predict_dataset(df_test)

    total_time = time.time() - start
    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_ram / (1024 * 1024)
    n_anomalies = (result["iso_label"] == "anomalie").sum()

    print(
        f"Iteration {i+1} | Samples: {len(df_test):>6} "
        f"| Time: {total_time:.4f}s "
        f"| Max RAM: {peak_ram_mb:.2f} MB "
        f"| Anomalies: {n_anomalies} ({100 * n_anomalies / len(df_test):.1f}%)"
    )
