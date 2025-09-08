import re
import matplotlib.pyplot as plt

# File log
log_file = "training_log.txt"

# Regex để parse từng dòng
pattern = re.compile(
    r"step (\d+):.*reward: ([\d\.\-]+), loss: ([\d\.\-]+), q: ([\d\.\-]+), avg_ep_reward: ([\d\.\-]+)"
)

steps, rewards, losses, q_scores, avg_rewards = [], [], [], [], []

# Đọc file và parse dữ liệu
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            rewards.append(float(match.group(2)))
            losses.append(float(match.group(3)))
            q_scores.append(float(match.group(4)))
            avg_rewards.append(float(match.group(5)))

# Vẽ 4 biểu đồ
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(steps, rewards, label="Reward per action", linewidth=0.8)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward per Action vs Step")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(steps, losses, color="red", label="Loss", linewidth=0.8)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss vs Step")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(steps, q_scores, color="green", label="Q Score", linewidth=0.8)
plt.xlabel("Step")
plt.ylabel("Q Score")
plt.title("Q Score vs Step")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(steps, avg_rewards, color="purple", label="Avg Episode Reward", linewidth=0.8)
plt.xlabel("Step")
plt.ylabel("Avg Episode Reward")
plt.title("Average Episode Reward vs Step")
plt.legend()

plt.tight_layout()
plt.show()
