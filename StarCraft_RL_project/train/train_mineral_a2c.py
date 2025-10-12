# train/train_mineral_a2c.py
import os
import sys
import yaml
import csv
import time
import numpy as np
import torch
from pysc2.lib import actions
from agents.a2c_agent import A2CAgent
from envs.mineral_env import MineralEnv
from absl import flags

# --- Fix path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- Parse flags ---
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(sys.argv)

# --- Load config ---
config_path = os.path.join(ROOT_DIR, "config/mineral_a2c.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- Init environment ---
env = MineralEnv(visualize=True)
input_shape = (1, 64, 64)
action_dim = 2  # 0=no_op, 1=move_to_mineral

# --- Init agent ---
agent = A2CAgent(
    input_shape=input_shape,
    n_actions=action_dim,
    lr=cfg['learning_rate'],
    gamma=cfg['gamma']
)

# --- Helper: map action index to pysc2 action ---
def map_action(action_idx, obs, unit):
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    avail = obs.observation["available_actions"]
    acts = []

    if select_id in avail:
        acts.append(actions.FUNCTIONS.select_point("select", [int(unit.x), int(unit.y)]))
        obs = env.step(acts)
        acts = []

    if action_idx == 0 or move_id not in avail:
        obs = env.step([actions.FUNCTIONS.no_op()])
    else:
        minerals = [m for m in obs.observation["feature_units"] if m.alliance == 3]
        if minerals:
            target = min(minerals, key=lambda m: (m.x - unit.x) ** 2 + (m.y - unit.y) ** 2)
            obs = env.step([actions.FUNCTIONS.Move_screen("now", [int(target.x), int(target.y)])])
        else:
            obs = env.step([actions.FUNCTIONS.no_op()])

    return obs

# --- Directories ---
model_dir = os.path.join(ROOT_DIR, "models/a2c")
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "a2c_mineral_rewards.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

# --- Training loop ---
num_episodes = cfg['num_episodes']
save_every = cfg.get('save_model_every', 50)

try:
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
            if not player_units:
                obs = env.step([actions.FUNCTIONS.no_op()])
                continue

            for unit in player_units:
                # --- Prepare state ---
                state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]

                # --- Select action ---
                action_idx, log_prob, value = agent.select_action(state)
                obs = map_action(action_idx, obs, unit)

                reward = torch.tensor([obs.reward], dtype=torch.float32)
                done = obs.last()
                next_state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]


                # --- Update step ---
                next_value = agent.model(torch.tensor(next_state, dtype=torch.float32).to(agent.device))[1]
                agent.update(log_prob, value, reward, next_value, done)

                total_reward += reward.item()
                if done:
                    break

        # --- Save model periodically ---
        if (ep + 1) % save_every == 0:
            path = os.path.join(model_dir, f"a2c_mineral_ep{ep+1}.pth")
            agent.save(path)
            print(f"💾 Saved model at {path}")

        # --- Log reward với 2 chữ số ---
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{total_reward:.2f}"])

        print(f"✅ Episode {ep+1}/{num_episodes} | Total Minerals (Reward) = {total_reward:.2f}")

finally:
    # --- Save final model ---
    final_path = os.path.join(model_dir, "a2c_mineral_final.pth")
    agent.save(final_path)
    print(f"💾 Final model saved at: {final_path}")
    env.close()
    time.sleep(1)
