# train/train_beacon_a2c.py
import os
import sys
import yaml
import numpy as np
import torch
import csv
from pysc2.lib import actions
from agents.a2c_agent import A2CAgent
from envs.beacon_env import BeaconEnv
from absl import flags
import time

# --- Fix path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- Parse flags ---
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(sys.argv)

# --- Load config ---
config_path = os.path.join(ROOT_DIR, "config/beacon_a2c.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- Init environment ---
env = BeaconEnv(visualize=True)
input_shape = (1, 64, 64)
n_actions = 2
agent = A2CAgent(
    input_shape,
    n_actions,
    lr=cfg['learning_rate'],
    gamma=cfg['gamma']
)

# --- Helper functions ---
def get_closest_target(unit, targets):
    return min(targets, key=lambda t: (t.x - unit.x) ** 2 + (t.y - unit.y) ** 2)

def map_action_multiunit(obs):
    avail = obs.observation['available_actions']
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
    targets = [u for u in obs.observation["feature_units"] if u.alliance == 3]

    if not player_units:
        return [actions.FUNCTIONS.no_op()]

    action_list = []
    for unit in player_units:
        if move_id in avail and targets:
            target = get_closest_target(unit, targets)
            action_list.append(actions.FUNCTIONS.Move_screen("now", [int(target.x), int(target.y)]))
        elif select_id in avail:
            action_list.append(actions.FUNCTIONS.select_point("select", [int(unit.x), int(unit.y)]))
        else:
            action_list.append(actions.FUNCTIONS.no_op())
    return action_list

# --- Prepare folders ---
model_dir = os.path.join(ROOT_DIR, "models/a2c")
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
reward_file = os.path.join(log_dir, "a2c_beacon_rewards.csv")

if not os.path.exists(reward_file):
    with open(reward_file, "w", newline="") as f:
        csv.writer(f).writerow(["Episode", "TotalReward"])

# --- Training loop ---
num_episodes = cfg['num_episodes']
reward_history = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.tensor(
            np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :],
            device=agent.device
        )

        # --- Select action ---
        action_idx, log_prob, value = agent.select_action(state)
        actions_to_step = map_action_multiunit(obs)
        next_obs = env.step(actions_to_step)

        # --- Reward from next_obs directly ---
        reward = torch.tensor([next_obs.reward], dtype=torch.float32, device=agent.device)
        done_tensor = torch.tensor([next_obs.last()], dtype=torch.float32, device=agent.device)

        next_state = torch.tensor(
            np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :],
            device=agent.device
        )
        with torch.no_grad():
            _, next_value = agent.model(next_state)

        # --- Update agent ---
        agent.update(log_prob, value, reward, next_value, done_tensor)

        total_reward += reward.item()
        obs = next_obs
        done = next_obs.last()

    # --- Log reward ---
    reward_history.append(total_reward)
    with open(reward_file, "a", newline="") as f:
        csv.writer(f).writerow([episode + 1, total_reward])

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward = {total_reward:.2f}")

# --- Save final model ---
final_model_path = os.path.join(model_dir, "a2c_beacon_final.pth")
agent.save(final_model_path)

print(f"✅ Saved final model to {final_model_path}")
print(f"✅ Saved reward history to {reward_file}")

env.close()
time.sleep(1)
