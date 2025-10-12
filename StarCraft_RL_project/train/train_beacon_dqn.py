import os
import sys
import yaml
import numpy as np
import torch
import csv
from pysc2.lib import actions
from agents.dqn_agent import DQNAgent
from envs.beacon_env import BeaconEnv
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
config_path = os.path.join(ROOT_DIR, "config/beacon_dqn.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- Init environment ---
env = BeaconEnv(visualize=False)
input_shape = (1, 64, 64)
n_actions = 2

# --- Init agent ---
agent = DQNAgent(
    input_shape=input_shape,
    n_actions=n_actions,
    lr=cfg["learning_rate"],
    gamma=cfg["gamma"],
    eps_start=cfg["eps_start"],
    eps_end=cfg["eps_end"],
    eps_decay=cfg["eps_decay"],
    batch_size=cfg["batch_size"]
)

# --- Helper: map action index to pysc2 action ---
def map_action(action_idx, obs):
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    available = obs.observation["available_actions"]

    if move_id not in available and select_id in available:
        player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
        if player_units:
            u = player_units[0]
            return actions.FUNCTIONS.select_point("select", [int(u.x), int(u.y)])
        return actions.FUNCTIONS.no_op()

    if action_idx == 0:
        return actions.FUNCTIONS.no_op()
    elif action_idx == 1:
        player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
        minerals = [u for u in obs.observation["feature_units"] if u.alliance == 3]
        if player_units and minerals:
            player = player_units[0]
            target = min(minerals, key=lambda u: (u.x - player.x) ** 2 + (u.y - player.y) ** 2)
            return actions.FUNCTIONS.Move_screen("now", [int(target.x), int(target.y)])
    return actions.FUNCTIONS.no_op()

# --- Prepare dirs ---
model_dir = os.path.join(ROOT_DIR, "models/dqn")
checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints/dqn")
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "dqn_beacon_rewards.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward"])

# --- Training loop ---
reward_history = []
update_target_every = cfg["update_target_every"]
save_model_every = cfg["save_model_every"]
num_episodes = cfg["num_episodes"]

try:
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # --- Lấy state ---
            state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]

            # --- Chọn action ---
            action_idx = agent.select_action(state)
            action = map_action(action_idx, obs)

            # --- Thực hiện bước ---
            next_obs = env.step(action)
            reward = next_obs.reward
            done = next_obs.last()
            next_state = np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]

            # --- Thêm penalty nếu đứng yên ---
            if action_idx == 0:
                reward -= 0.1  # penalty khi no_op

            # --- Lưu transition vào memory ---
            agent.store_transition(state, action_idx, reward, next_state, done)

            total_reward += reward
            obs = next_obs

            # --- Cập nhật batch sau mỗi bước với xác suất ---
            if len(agent.memory) >= agent.batch_size and np.random.rand() < 0.25:
                agent.update()

        # --- Cập nhật target định kỳ ---
        if (episode + 1) % update_target_every == 0:
            agent.update_target()

        # --- Lưu checkpoint ---
        if (episode + 1) % save_model_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dqn_beacon_ep{episode+1}.pth")
            torch.save(agent.policy_net.state_dict(), ckpt_path)

        # --- Ghi log ---
        reward_history.append(total_reward)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])

        print(f" Episode {episode+1}/{num_episodes} | Reward={total_reward:.2f}")

finally:
    # --- Save final model ---
    final_path = os.path.join(model_dir, "dqn_beacon_final.pth")
    agent.save(final_path)
    print(f"💾 Final model saved to {final_path}")
    env.close()