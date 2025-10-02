# test_ppo_final.py
import gym
import numpy as np
from stable_baselines3 import PPO
from sc2_gym_wrapper import MoveToBeaconWrapper  # Wrapper bạn đã làm
import time

# ----------------------------
# Cấu hình
# ----------------------------
MODEL_PATH = "ppo_move_to_beacon_final.zip"  # model đã train
NUM_EPISODES = 5
SLEEP_BETWEEN_STEPS = 0.05  # Nếu visualize, delay để nhìn agent

# ----------------------------
# Tạo environment
# ----------------------------
env = MoveToBeaconWrapper()
env.env.render = True  # bật hiển thị PySC2

# ----------------------------
# Load model
# ----------------------------
model = PPO.load(MODEL_PATH)

# ----------------------------
# Test agent
# ----------------------------
for ep in range(1, NUM_EPISODES + 1):
    obs = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        time.sleep(SLEEP_BETWEEN_STEPS)  # delay để dễ nhìn

    print(f"=== Episode {ep} finished after {step_count} steps. Total reward = {total_reward} ===")

env.close()
