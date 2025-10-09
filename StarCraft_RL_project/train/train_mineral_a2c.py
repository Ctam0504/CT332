# train/train_mineral_a2c_real_reward.py
import os, sys, yaml, numpy as np, torch, time, csv
from pysc2.lib import actions
from envs.mineral_env import MineralEnv
from agents.a2c_agent import A2CAgent
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

# --- Init environment and agent ---
env = MineralEnv(visualize=True)
input_shape = (1, 64, 64)
n_actions = 2
agent = A2CAgent(input_shape, n_actions,
                 lr=cfg['learning_rate'], gamma=cfg['gamma'])

# --- Helper functions ---
def get_mineral_pos(obs):
    units = obs.observation["feature_units"]
    minerals = [u for u in units if u.alliance == 3]
    if not minerals:
        return None
    player_units = [u for u in units if u.alliance == 1]
    if not player_units:
        return None
    agent_unit = player_units[0]
    distances = [((u.x - agent_unit.x)**2 + (u.y - agent_unit.y)**2, u) for u in minerals]
    closest = min(distances, key=lambda x: x[0])[1]
    return [int(closest.x), int(closest.y)]

def map_action(obs):
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id
    avail = obs.observation['available_actions']
    action_list = []

    player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]
    minerals = [u for u in obs.observation["feature_units"] if u.alliance == 3]

    if not player_units:
        return [actions.FUNCTIONS.no_op()]

    for unit in player_units:
        if move_id in avail and minerals:
            target = min(minerals, key=lambda m: (m.x-unit.x)**2 + (m.y-unit.y)**2)
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
log_file = os.path.join(log_dir, "a2c_mineral_rewards.csv")

# --- Write header if not exist ---
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

# --- Training ---
num_episodes = cfg['num_episodes']

try:
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
            actions_to_step = map_action(obs)
            next_obs = env.step(actions_to_step)

            # --- Reward thực tế từ môi trường ---
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

        # --- Log reward per episode ---
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([episode + 1, total_reward])

        print(f"✅ Episode {episode + 1}/{num_episodes} | Total Reward = {total_reward:.2f}")

finally:
    print("✅ Training finished! Saving final A2C model...")
    model_path = os.path.join(model_dir, "a2c_mineral_final.pth")
    agent.save(model_path)
    print(f"💾 Final model saved at: {model_path}")
    print("Closing environment...")
    env.close()
    time.sleep(1)
