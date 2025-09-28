# env_setup.py
# Script test PySC2 minigames, log reward cho Reinforcement Learning

import os
import csv
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import pandas as pd
import matplotlib.pyplot as plt
from absl import flags

# --------------------------
# Fix lỗi absl.flags chưa parse
# --------------------------
FLAGS = flags.FLAGS
FLAGS(["env_setup"])  # giả lập argv để tránh lỗi

# --------------------------
# Cấu hình map và CSV
# --------------------------
MAP_NAME = "MoveToBeacon"   # có thể đổi sang CollectMineralShards
CSV_FILE = "../data/results.csv"
MAX_EPISODES = 10           # thường train cần 1000+, ở đây chỉ demo
STEP_MUL = 8
SCREEN_SIZE = 64
VISUALIZE = False           # True để hiện cửa sổ StarCraft (nặng máy)

# Tạo folder data nếu chưa có
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Tạo file CSV với header
with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "step", "step_reward", "total_reward"])

# --------------------------
# Khởi tạo môi trường
# --------------------------
env = sc2_env.SC2Env(
    map_name=MAP_NAME,
    players=[sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=SCREEN_SIZE, minimap=SCREEN_SIZE),
        use_feature_units=True,
    ),
    step_mul=STEP_MUL,
    game_steps_per_episode=0,
    visualize=VISUALIZE,
)

# --------------------------
# Hàm chọn action ngẫu nhiên hợp lệ
# --------------------------
def random_move_action(obs):
    """
    Agent chọn một hành động Move_screen nếu khả dụng,
    di chuyển tới tọa độ random trên màn hình.
    """
    available_actions = obs.observation["available_actions"]
    if actions.FUNCTIONS.Move_screen.id in available_actions:
        x = np.random.randint(0, SCREEN_SIZE)
        y = np.random.randint(0, SCREEN_SIZE)
        return actions.FUNCTIONS.Move_screen("now", [x, y])
    else:
        return actions.FUNCTIONS.no_op()

# --------------------------
# Vòng lặp episode
# --------------------------
for episode in range(1, MAX_EPISODES + 1):
    obs = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    while not done:
        step_count += 1
        action = random_move_action(obs[0])
        obs = env.step([action])
        step_reward = obs[0].reward
        total_reward += step_reward
        done = obs[0].last()

        # Ghi log vào CSV
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, step_count, step_reward, total_reward])

    print(f"Episode {episode} finished. Total reward: {total_reward:.2f}")

env.close()
print(f"All episodes done. Rewards saved to {CSV_FILE}")

# --------------------------
# Vẽ biểu đồ reward theo episode
# --------------------------
df = pd.read_csv(CSV_FILE)

# Lấy reward cuối mỗi episode (reward tổng)
episode_rewards = df.groupby("episode")["total_reward"].max()

plt.plot(episode_rewards.index, episode_rewards.values, marker="o")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"Reward over Episodes ({MAP_NAME})")
plt.grid(True)
plt.savefig("../data/reward_plot.png")
plt.show()
