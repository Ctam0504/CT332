# train_mineral_dqn_real_reward_fix.py
import os
import sys
import yaml
import numpy as np
import time
import csv
from pysc2.lib import actions
from agents.dqn_agent import DQNAgent
from envs.mineral_env import MineralEnv
from absl import flags

# --- Fix path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- Parse absl flags ---
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(sys.argv)

# --- Load config ---
config_path = os.path.join(ROOT_DIR, "config/mineral_dqn.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- Init environment ---
env = MineralEnv(visualize=True)
input_shape = (1, 64, 64)
n_actions = 2  # 0=no_op, 1=move

agent = DQNAgent(
    input_shape, n_actions,
    lr=cfg['learning_rate'],
    gamma=cfg['gamma'],
    eps_start=cfg['eps_start'],
    eps_end=cfg['eps_end'],
    eps_decay=cfg['eps_decay'],
    batch_size=cfg['batch_size']
)

# --- Helper functions ---
def get_mineral_pos(obs, unit):
    units = obs.observation["feature_units"]
    minerals = [u for u in units if u.alliance == 3]
    if not minerals:
        return None
    distances = [((m.x - unit.x)**2 + (m.y - unit.y)**2, m) for m in minerals]
    closest = min(distances, key=lambda x: x[0])[1]
    return [int(closest.x), int(closest.y)]

def move_unit(obs, unit):
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    avail = obs.observation["available_actions"]
    actions_to_take = []

    # Select unit
    if select_id in avail:
        actions_to_take.append(actions.FUNCTIONS.select_point("select", [int(unit.x), int(unit.y)]))
        obs = env.step(actions_to_take)
        actions_to_take = []

    # Get state
    state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]
    action_idx = agent.select_action(state)

    # Move to mineral if action = 1
    if action_idx == 1 and move_id in avail:
        target = get_mineral_pos(obs, unit)
        if target:
            obs = env.step([actions.FUNCTIONS.Move_screen("now", target)])
        else:
            obs = env.step([actions.FUNCTIONS.no_op()])
    else:
        obs = env.step([actions.FUNCTIONS.no_op()])

    return obs, state, action_idx

# --- Prepare directories ---
model_dir = os.path.join(ROOT_DIR, "models/dqn")
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "dqn_mineral_rewards_real_fix.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])

# --- Training loop ---
num_episodes = cfg['num_episodes']

try:
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
            if not player_units:
                obs = env.step([actions.FUNCTIONS.no_op()])
                continue

            for unit in player_units:
                next_obs, state, action_idx = move_unit(obs, unit)

                # Reward = số mineral thực tế
                reward = next_obs.reward

                done = next_obs.last()
                next_state = np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]

                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.update()

                total_reward += reward
                obs = next_obs

                if done:
                    break

        # --- Log reward per episode ---
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])

        # Update target network
        if (episode + 1) % cfg['update_target_every'] == 0:
            agent.update_target()

        print(f"✅ Episode {episode+1}/{num_episodes} | Total Minerals (Reward) = {total_reward:.2f}")

finally:
    print("✅ Training finished! Saving final model...")
    final_model_path = os.path.join(model_dir, "dqn_mineral_final.pth")
    agent.save(final_model_path)
    print(f"💾 Final model saved at: {final_model_path}")
    print("Closing environment...")
    env.close()
    time.sleep(1)
