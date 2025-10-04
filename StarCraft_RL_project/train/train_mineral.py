import os
import sys
import yaml
import numpy as np
from agents.dqn_agent import DQNAgent
from envs.mineral_env import MineralEnv
from pysc2.lib import actions
import time

# --- fix path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- parse absl flags ---
from absl import flags
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(sys.argv)

# --- load config ---
config_path = os.path.join(ROOT_DIR, "config/mineral_config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- init env ---
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

# --- helper functions ---
def get_mineral_pos(obs, unit):
    """Lấy mineral gần nhất từ 1 unit"""
    units = obs.observation["feature_units"]
    minerals = [u for u in units if u.alliance == 3]
    if not minerals:
        return None
    distances = [((m.x - unit.x)**2 + (m.y - unit.y)**2, m) for m in minerals]
    closest = min(distances, key=lambda x: x[0])[1]
    return [int(closest.x), int(closest.y)]

def map_actions_multi(obs):
    """Map action DQN → pysc2 action riêng cho từng unit"""
    move_id = actions.FUNCTIONS.Move_screen.id
    select_id = actions.FUNCTIONS.select_point.id

    player_units = [u for u in obs.observation["feature_units"] if u.alliance == 1]

    actions_list = []
    for unit in player_units:
        # state riêng cho unit
        state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]
        action_idx = agent.select_action(state)

        # select unit
        if select_id in obs.observation['available_actions']:
            actions_list.append(actions.FUNCTIONS.select_point("select", [int(unit.x), int(unit.y)]))

        # move unit
        if action_idx == 1 and move_id in obs.observation['available_actions']:
            target = get_mineral_pos(obs, unit)
            if target is not None:
                actions_list.append(actions.FUNCTIONS.Move_screen("now", target))

    if not actions_list:
        actions_list.append(actions.FUNCTIONS.no_op())

    return actions_list

def step_multi(env, actions_list):
    obs = None
    for act in actions_list:
        obs = env.step(act)
    return obs

# --- training loop ---
num_episodes = cfg['num_episodes']

try:
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # map actions cho tất cả units
            actions_list = map_actions_multi(obs)
            # step env nhiều lần
            next_obs = step_multi(env, actions_list)

            reward = next_obs.reward
            done = next_obs.last()

            # state/next_state cho DQN
            state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]
            next_state = np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]

            # lưu transition và update
            agent.store_transition(state, 1 if len(actions_list)>1 else 0, reward, next_state, done)
            agent.update()

            total_reward += reward

            # debug print
            player_units_pos = [(u.x, u.y) for u in obs.observation["feature_units"] if u.alliance == 1]
            minerals_pos = [(u.x, u.y) for u in obs.observation["feature_units"] if u.alliance == 3]
            print(f"Step {step_count}: reward={reward}, player={player_units_pos}, mineral={minerals_pos}")

            obs = next_obs
            step_count += 1

        # update target network
        if (episode + 1) % cfg['update_target_every'] == 0:
            agent.update_target()

        print(f"Episode {episode+1}/{num_episodes}, Total Reward={total_reward}")

finally:
    print("Closing environment...")
    env.close()
    time.sleep(1)
