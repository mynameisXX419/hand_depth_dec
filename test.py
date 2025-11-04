import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/ljy/project/hand_dec/ljy/ljy_1/pressure_log.csv")
plt.plot(df["time_ms"], df["press_sum_norm"])
plt.xlabel("time (ms)")
plt.ylabel("press_sum_norm")
plt.show()
