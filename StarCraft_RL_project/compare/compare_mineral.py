import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Đường dẫn gốc của project ---
project_root = os.path.dirname(os.path.abspath(__file__))  # thư mục compare/
project_root = os.path.dirname(project_root)  # lùi lên 1 cấp -> starcraft_rl_project

# --- Đường dẫn file log ---
dqn_file = os.path.join(project_root, "logs", "dqn_mineral_rewards.csv")
a2c_file = os.path.join(project_root, "logs", "a2c_mineral_rewards.csv")

# --- Hàm lấy cột reward dù tên khác nhau ---
def get_reward_col(df):
    for col in ["reward", "Reward", "total_reward", "TotalReward"]:
        if col in df.columns:
            return df[col]
    raise ValueError(f"Không tìm thấy cột reward trong {df.columns}")

# --- Đọc dữ liệu ---
dqn_data = pd.read_csv(dqn_file)
a2c_data = pd.read_csv(a2c_file)

# --- Lấy cột reward ---
dqn_rewards = get_reward_col(dqn_data)
a2c_rewards = get_reward_col(a2c_data)

print("===== Thống kê Reward (Mineral Map) =====")
print(f"DQN: trung bình = {dqn_rewards.mean():.2f}, max = {dqn_rewards.max()}, min = {dqn_rewards.min()}")
print(f"A2C: trung bình = {a2c_rewards.mean():.2f}, max = {a2c_rewards.max()}, min = {a2c_rewards.min()}")

# --- Tạo thư mục lưu biểu đồ ---
output_dir = os.path.join(project_root, "compare_reward")
os.makedirs(output_dir, exist_ok=True)

# --- Vẽ biểu đồ ---
plt.figure(figsize=(10, 5))
plt.plot(dqn_rewards, label="DQN - Mineral", color="blue")
plt.plot(a2c_rewards, label="A2C - Mineral", color="green")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("So sánh Reward - Map Mineral")
plt.legend()
plt.grid(True)

# --- Lưu file ---
output_file = os.path.join(output_dir, "compare_mineral.png")
plt.savefig(output_file)
plt.close()

print(f"✅ Đã lưu biểu đồ tại {output_file}")
