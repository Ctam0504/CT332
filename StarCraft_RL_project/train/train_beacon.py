# train/train_beacon.py

import os
import sys
import yaml
import numpy as np
from agents.dqn_agent import DQNAgent
from envs.beacon_env import BeaconEnv
from pysc2.lib import actions

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
config_path = os.path.join(ROOT_DIR, "config/beacon_config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# --- init env ---
env = BeaconEnv()
input_shape = (1, 64, 64)  # 1 channel, 64x64 screen
n_actions = 2               # 0=no_op, 1=move
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
def get_beacon_pos(obs):
    screen = obs.observation["feature_screen"]["player_relative"]
    ys, xs = np.where(screen == 3)  # 3 = beacon
    if len(xs) > 0:
        return [int(xs.mean()), int(ys.mean())]
    return None

def map_action(action_idx, obs):
    # Force select unit if Move_screen not available
    if actions.FUNCTIONS.Move_screen.id not in obs.observation['available_actions']:
        if actions.FUNCTIONS.select_point.id in obs.observation['available_actions']:
            # select player unit
            player_y, player_x = np.where(obs.observation["feature_screen"]["player_relative"] == 1)
            if len(player_x) > 0:
                target = [int(player_x.mean()), int(player_y.mean())]
                return actions.FUNCTIONS.select_point("select", target)
        # fallback noop
        return actions.FUNCTIONS.no_op()

    # map DQN action
    if action_idx == 0:
        return actions.FUNCTIONS.no_op()
    elif action_idx == 1:
        target = get_beacon_pos(obs)
        if target is not None:
            return actions.FUNCTIONS.Move_screen("now", target)
    return actions.FUNCTIONS.no_op()

# --- training loop ---
num_episodes = cfg['num_episodes']
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # prepare state
        state = np.array(obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]
        # select action
        action_idx = agent.select_action(state)
        # map to pysc2 action
        action = map_action(action_idx, obs)
        # step
        next_obs = env.step(action)
        reward = next_obs.reward
        done = next_obs.last()
        # prepare next_state
        next_state = np.array(next_obs.observation["feature_screen"]["player_relative"], dtype=np.float32)[None, None, :, :]
        # store transition
        agent.store_transition(state, action_idx, reward, next_state, done)
        # update DQN
        agent.update()
        # update total reward
        total_reward += reward
        # debug print
        beacon_pos = get_beacon_pos(obs)
        print(f"Step {step_count}: action={action_idx}, reward={reward}, beacon={beacon_pos}, available_actions={obs.observation['available_actions']}")
        obs = next_obs
        step_count += 1

    # update target network
    if (episode + 1) % cfg['update_target_every'] == 0:
        agent.update_target()

    print(f"Episode {episode+1}/{num_episodes}, Total Reward={total_reward}")

env.close()
