# src/generate_data.py

import pandas as pd
import numpy as np

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

df.to_csv("data/telemetry.csv",index=False)

print(df.head())