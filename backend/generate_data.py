from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telemetry.csv"

np.random.seed(42)

rows = 10000

cpu = np.random.uniform(0,100,rows)
memory = np.random.uniform(0,100,rows)
disk = np.random.uniform(0,100,rows)
temperature = np.random.uniform(20,95,rows)
network = np.random.uniform(0,1000,rows)

failure = (
    (cpu > 90) &
    (memory > 85) &
    (temperature > 80)
).astype(int)

df = pd.DataFrame({
    "cpu": cpu,
    "memory": memory,
    "disk": disk,
    "temperature": temperature,
    "network": network,
    "failure": failure
})

df.to_csv(DATA_PATH, index=False)

print(df.head())
