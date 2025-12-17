import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pressure_series.csv")

plt.figure(figsize=(10,4))
plt.plot(df["host_ms"], df["val_raw"], label="raw", alpha=0.4)
plt.plot(df["host_ms"], df["val_filt"], label="filtered", linewidth=2)
plt.legend()
plt.xlabel("time (ms)")
plt.ylabel("pressure")
plt.title("Pressure signal: raw vs filtered")
plt.show()
