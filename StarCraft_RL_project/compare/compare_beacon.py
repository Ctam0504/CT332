import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Đường dẫn gốc project ---
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)  # lên 1 cấp -> StarCraft_RL_project

# --- Đường dẫn file log ---

dqn_file = os.path.join(project_root, "logs", "dqn_beacon_rewards.csv")
a2c_file = os.path.join(project_root, "logs", "a2c_beacon_rewards.csv")
ppo_file = os.path.join(project_root, "logs", "ppo_beacon_rewards.csv")

# --- Hàm lấy cột reward dù tên khác nhau ---

def get_reward_col(df):
    for col in ["reward", "Reward", "total_reward", "TotalReward"]:
        if col in df.columns:
            return df[col]
    raise ValueError(f"Không tìm thấy cột reward trong {df.columns}")

# --- Đọc dữ liệu ---

print("📂 Đang đọc dữ liệu log...")
dqn_data = pd.read_csv(dqn_file)
a2c_data = pd.read_csv(a2c_file)
ppo_data = pd.read_csv(ppo_file)

# --- Lấy cột reward ---

dqn_rewards = get_reward_col(dqn_data)
a2c_rewards = get_reward_col(a2c_data)
ppo_rewards = get_reward_col(ppo_data)

# --- Thống kê ---

print("===== Thống kê Reward =====")
print(f"DQN: mean = {dqn_rewards.mean():.2f}, max = {dqn_rewards.max():.2f}, min = {dqn_rewards.min():.2f}")
print(f"A2C: mean = {a2c_rewards.mean():.2f}, max = {a2c_rewards.max():.2f}, min = {a2c_rewards.min():.2f}")
print(f"PPO: mean = {ppo_rewards.mean():.2f}, max = {ppo_rewards.max():.2f}, min = {ppo_rewards.min():.2f}")

# --- Tạo thư mục lưu biểu đồ ---

output_dir = os.path.join(project_root, "compare_reward")
os.makedirs(output_dir, exist_ok=True)

# --- Vẽ biểu đồ ---

plt.figure(figsize=(10, 5))
plt.plot(dqn_rewards, label="DQN", color="blue", linewidth=1.5)
plt.plot(a2c_rewards, label="A2C", color="green", linewidth=1.5)
plt.plot(ppo_rewards, label="PPO", color="orange", linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("So sánh Reward - Map Beacon")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# --- Lưu file ---

output_file = os.path.join(output_dir, "compare_beacon.png")
plt.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close()

print(f"✅ Đã lưu biểu đồ tại {output_file}")
