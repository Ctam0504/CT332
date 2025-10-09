import os
import sys
import yaml
import csv
import time
import torch
import numpy as np
from absl import flags
from pysc2.lib import actions

from agents.ppo_agent import PPOAgent
from envs.beacon_env import BeaconEnv

# --- Setup path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(sys.argv)

# --- Load config ---
with open(os.path.join(ROOT_DIR, "config/beacon_ppo.yaml")) as f:
    cfg = yaml.safe_load(f)

# --- Init environment ---
env = BeaconEnv(visualize=False)
obs_dim = 64 * 64
n_actions = 2  # no_op, move_to_beacon

agent = PPOAgent(
    obs_dim=obs_dim,
    n_actions=n_actions,
    lr=cfg['learning_rate'],
    gamma=cfg['gamma'],
    clip_range=cfg['clip_range'],
    gae_lambda=cfg['gae_lambda'],
    entropy_coef=cfg['entropy_coef'],
    value_loss_coef=cfg['value_loss_coef'],
    batch_size=cfg['batch_size'],
    update_epochs=cfg['update_epochs'],
    hidden_size=cfg['hidden_size']
)

# --- Setup output folders ---
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("checkpoints/ppo", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_file = "logs/ppo_beacon_rewards.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Episode", "TotalReward"])

# --- Helper function ---
def map_action(action_idx, obs):
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    available = obs.observation['available_actions']

    if move_id not in available and select_id in available:
        player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
        if player_units:
            u = player_units[0]
            return actions.FUNCTIONS.select_point("select", [int(u.x), int(u.y)])
        return actions.FUNCTIONS.no_op()

    if action_idx == 0:
        return actions.FUNCTIONS.no_op()
    if action_idx == 1:
        player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
        beacons = [u for u in obs.observation["feature_units"] if u.alliance == 3]
        if player_units and beacons:
            player = player_units[0]
            target = min(beacons, key=lambda u: (u.x - player.x)**2 + (u.y - player.y)**2)
            return actions.FUNCTIONS.Move_screen("now", [int(target.x), int(target.y)])
    return actions.FUNCTIONS.no_op()

# --- Training loop ---
reward_history = []
trajectory = []

try:
    for episode in range(cfg['num_episodes']):
        obs = env.reset()
        done = False
        total_reward = 0
        trajectory.clear()

        while not done:
            state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)
            action_idx, log_prob, value = agent.select_action(state)
            action = map_action(action_idx, obs)
            next_obs = env.step(action)
            reward = next_obs.reward
            done = next_obs.last()
            next_state = np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)

            trajectory.append((
    state,
    action_idx,
    log_prob.detach().unsqueeze(0),
    value.detach().unsqueeze(0),
    reward,
    next_state,
    done
))

            total_reward += reward
            obs = next_obs

        agent.update(trajectory)
        reward_history.append(total_reward)

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([episode + 1, total_reward])

        if (episode + 1) % cfg['save_model_every'] == 0:
            torch.save(agent.state_dict(), f"checkpoints/ppo/ppo_beacon_ep{episode+1}.pth")

        print(f"✅ Episode {episode+1}/{cfg['num_episodes']} | Total Reward = {total_reward:.2f}")

finally:
    final_path = "models/ppo/ppo_beacon_final.pth"
    agent.save(final_path)
    print(f"💾 Final model saved at: {final_path}")
    env.close()
    time.sleep(1)
