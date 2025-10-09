# train/train_mineral_ppo_real_reward.py
import os
import sys
import yaml
import numpy as np
import torch
import csv
import time
from pysc2.lib import actions
from envs.mineral_env import MineralEnv
from agents.ppo_agent import PPOAgent
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
config_path = os.path.join(ROOT_DIR, "config/mineral_ppo.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- Init environment and agent ---
env = MineralEnv(visualize=True)
obs_dim = 64 * 64
action_dim = 2  # 0=no_op, 1=move

agent = PPOAgent(
    obs_dim=obs_dim,
    n_actions=action_dim,
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
    acts = []

    if select_id in avail:
        acts.append(actions.FUNCTIONS.select_point("select", [int(unit.x), int(unit.y)]))
        obs = env.step(acts)
        acts = []

    state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32).flatten()
    action, log_prob, _ = agent.select_action(state)

    if action == 1 and move_id in avail:
        target = get_mineral_pos(obs, unit)
        if target:
            obs = env.step([actions.FUNCTIONS.Move_screen("now", target)])
        else:
            obs = env.step([actions.FUNCTIONS.no_op()])
    else:
        obs = env.step([actions.FUNCTIONS.no_op()])

    return obs, state, action, log_prob

# --- Directories ---
model_dir = os.path.join(ROOT_DIR, "models/ppo")
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "ppo_mineral_rewards.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

# --- Training ---
num_episodes = cfg['num_episodes']
save_every = cfg['save_model_every']

try:
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        states, log_probs, rewards = [], [], []

        while not done:
            player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
            if not player_units:
                obs = env.step([actions.FUNCTIONS.no_op()])
                continue

            for unit in player_units:
                next_obs, state, action, log_prob = move_unit(obs, unit)

                # <- reward thực tế từ môi trường
                reward = torch.tensor([next_obs.reward], dtype=torch.float32)
                done = next_obs.last()

                states.append(state)
                log_probs.append(log_prob)
                rewards.append(reward.item())

                total_reward += reward.item()
                obs = next_obs
                if done:
                    break

        # --- PPO update ---
        agent.optimizer.zero_grad()
        states_tensor = torch.FloatTensor(np.vstack(states))
        values = agent.evaluate(states_tensor).squeeze()
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + cfg['gamma'] * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        advantages = returns - values.detach()

        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        value_loss = (returns - values).pow(2).mean()
        loss = policy_loss + cfg['value_loss_coef'] * value_loss
        loss.backward()
        agent.optimizer.step()

        # --- Logging ---
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, total_reward])

        if (ep + 1) % save_every == 0:
            path = os.path.join(model_dir, f"ppo_mineral_ep{ep+1}.pth")
            torch.save(agent.state_dict(), path)
            print(f"💾 Saved model at {path}")

        print(f"✅ Episode {ep+1}/{num_episodes} | Total Minerals (Reward) = {total_reward:.2f}")

finally:
    print("✅ Training finished! Saving final model...")
    final_path = os.path.join(model_dir, "ppo_mineral_final.pth")
    torch.save(agent.state_dict(), final_path)
    print(f"💾 Final model saved at: {final_path}")
    print("Closing environment...")
    env.close()
    time.sleep(1)
