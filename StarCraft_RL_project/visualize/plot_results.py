import pandas as pd
import matplotlib.pyplot as plt
import os

# Thư mục lưu logs & hình
LOG_DIR = r"D:\CT332\StarCraft_RL_project\logs"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_and_save(file_path, title, output_name):
    df = pd.read_csv(file_path)

    required_cols = {'Agent', 'MeanReward', 'MaxReward', 'MinReward'}
    if not required_cols.issubset(df.columns):
        print(f"⚠ File '{file_path}' không có đủ cột {required_cols} để vẽ.")
        return

    agents = df['Agent']
    mean_rewards = df['MeanReward']
    max_rewards = df['MaxReward']
    min_rewards = df['MinReward']

    plt.figure(figsize=(10, 6))

    x = range(len(agents))
    plt.bar(x, mean_rewards, width=0.3, label='Mean Reward')
    plt.bar([p + 0.3 for p in x], max_rewards, width=0.3, label='Max Reward')
    plt.bar([p + 0.6 for p in x], min_rewards, width=0.3, label='Min Reward')

    plt.xticks([p + 0.3 for p in x], agents)
    plt.title(title)
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Lưu file ảnh PNG
    output_path = os.path.join(PLOT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ → {output_path}")

    plt.close()  # đóng figure để tiết kiệm bộ nhớ

def load_flexible_csv(file_path):
   
    df = pd.read_csv(file_path)

    # Chuẩn hóa tên cột về chữ thường để dễ kiểm tra
    lower_cols = [col.lower() for col in df.columns]
    df.columns = lower_cols

    # Đổi tên cột nếu trùng
    if 'episode' in df.columns and 'totalreward' in df.columns:
        df.rename(columns={'episode': 'Episode', 'totalreward': 'TotalReward'}, inplace=True)
    elif 'episode' in df.columns and 'total_reward' in df.columns:
        df.rename(columns={'episode': 'Episode', 'total_reward': 'TotalReward'}, inplace=True)
    else:
        print(f"⚠ Lỗi: File {file_path} không có cột Episode & Reward hợp lệ!")
        print("Cột tìm thấy:", df.columns.tolist())
        return None

    return df


def plot_single_agent(agent_name, map_name, file_name):
    file_path = os.path.join(LOG_DIR, file_name)
    df = load_flexible_csv(file_path)
    if df is None:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df['Episode'], df['TotalReward'])
    plt.title(f"{agent_name} - Map {map_name}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.4)

    output_path = os.path.join(PLOT_DIR, f"{agent_name}_{map_name}_reward.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu biểu đồ → {output_path}")


if __name__ == "__main__":
    # 📌 Vẽ cho từng map
    plot_and_save(
        os.path.join(LOG_DIR, "evaluate_beacon.csv"),
        "So sánh Reward của các Agent trên map Beacon",
        "beacon_rewards.png"
    )

    plot_and_save(
        os.path.join(LOG_DIR, "evaluate_mineral.csv"),
        "So sánh Reward của các Agent trên map Mineral",
        "mineral_rewards.png"
    )

    # 📌 Vẽ file tổng hợp nếu có
    all_map_file = os.path.join(LOG_DIR, "evaluate_all_maps.csv")
    if os.path.exists(all_map_file):
        plot_and_save(all_map_file, "So sánh Reward (Tổng hợp Beacon + Mineral)", "all_maps_rewards.png")


    single_plots = [
    ("A2C", "Beacon", "a2c_beacon_rewards.csv"),
    ("DQN", "Beacon", "dqn_beacon_rewards.csv"),
    ("PPO", "Beacon", "ppo_beacon_rewards.csv"),
    ("A2C", "Mineral", "a2c_mineral_rewards.csv"),
    ("DQN", "Mineral", "dqn_mineral_rewards.csv"),
    ("PPO", "Mineral", "ppo_mineral_rewards.csv"),
]

for agent, map_name, file_name in single_plots:
    plot_single_agent(agent, map_name, file_name)
