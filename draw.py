import pandas as pd
import matplotlib.pyplot as plt

# Đọc file, dùng regex để tách theo nhiều dấu space
df = pd.read_csv("results_Breakout_50M.txt", sep=r"\s+")

# Trích số step từ tên model (model_xxxxx.pt → xxxxx)
df["Step"] = df["Model"].str.extract(r"(\d+)").astype(int)

# Sắp xếp theo Step
df = df.sort_values("Step")

# Vẽ biểu đồ
plt.figure(figsize=(14,7))
plt.plot(df["Step"], df["Best"], label="Best Reward", linewidth=2)
plt.plot(df["Step"], df["Average"], label="Average Reward", linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.title("Best & Average Reward over Training Steps")
plt.legend()
plt.grid(True)
plt.show()
