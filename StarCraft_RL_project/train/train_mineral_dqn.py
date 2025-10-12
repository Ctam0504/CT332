# train/train_mineral_dqn_multi.py
import os
import sys
import yaml
import numpy as np
import torch
import csv
from pysc2.lib import actions
from agents.dqn_agent import DQNAgent
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
config_path = os.path.join(ROOT_DIR, "config/mineral_dqn.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- Init environment ---
env = MineralEnv(visualize=True)
input_shape = (1, 64, 64)
n_actions = 2  # 0=no_op, 1=move_to_mineral

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

# --- Helper: map action index to pysc2 action for one unit ---
def map_action_unit(action_idx, unit, obs):
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    avail = obs.observation["available_actions"]

    acts = []

    # Nếu unit chưa được chọn
    if select_id in avail:
        acts.append(actions.FUNCTIONS.select_point("select", [int(unit.x), int(unit.y)]))
        obs = env.step(acts)
        acts = []

    # Thực hiện action
    if action_idx == 0:
        acts.append(actions.FUNCTIONS.no_op())
    elif action_idx == 1 and move_id in avail:
        minerals = [m for m in obs.observation["feature_units"] if m.alliance == 3]
        if minerals:
            target = min(minerals, key=lambda m: (m.x - unit.x)**2 + (m.y - unit.y)**2)
            acts.append(actions.FUNCTIONS.Move_screen("now", [int(target.x), int(target.y)]))

    # Nếu không có action nào khả dụng, mặc định no_op
    if not acts:
        acts.append(actions.FUNCTIONS.no_op())

    return acts[0]

# --- Prepare dirs ---
model_dir = os.path.join(ROOT_DIR, "models/dqn")
checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints/dqn")
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "dqn_mineral_multi_rewards.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Episode", "TotalReward"])

# --- Training loop ---
reward_history = []
update_target_every = cfg["update_target_every"]
save_model_every = cfg["save_model_every"]
num_episodes = cfg["num_episodes"]
penalty = cfg.get("penalty", -0.05)

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

            # Mỗi unit chọn action riêng
            for unit in player_units[:2]:  # chỉ lấy 2 unit
                state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]
                action_idx = agent.select_action(state)
                action = map_action_unit(action_idx, unit, obs)

                # Thực hiện step
                next_obs = env.step([action])
                reward = next_obs.reward
                done = next_obs.last()
                next_state = np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]

                # Penalty nếu đứng yên
                if action_idx == 0:
                    reward += penalty

                # Lưu vào memory
                agent.store_transition(state, action_idx, reward, next_state, done)

                total_reward += reward
                obs = next_obs

                # Cập nhật batch ngẫu nhiên
                if len(agent.memory) >= agent.batch_size and np.random.rand() < 0.25:
                    agent.update()

        # --- Cập nhật target định kỳ ---
        if (episode + 1) % update_target_every == 0:
            agent.update_target()

        # --- Lưu checkpoint ---
        if (episode + 1) % save_model_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dqn_mineral_multi_ep{episode+1}.pth")
            torch.save(agent.policy_net.state_dict(), ckpt_path)

        # --- Ghi log ---
        reward_history.append(total_reward)
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([episode + 1, round(total_reward, 2)])

        print(f" Episode {episode+1}/{num_episodes} | Reward={total_reward:.2f}")

finally:
    final_path = os.path.join(model_dir, "dqn_mineral_multi_final.pth")
    agent.save(final_path)
    print(f"💾 Final model saved to {final_path}")
    env.close()
