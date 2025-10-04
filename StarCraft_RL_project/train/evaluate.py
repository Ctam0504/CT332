# ===================== train/evaluate.py =====================
import torch
import yaml
from agents.a2c_agent import A2CAgent
from envs.beacon_env import BeaconEnv
from envs.mineral_env import MineralEnv

def evaluate(agent, env, num_episodes=5):
    total_rewards = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            state = obs.observation['feature_screen']['player_relative'][None, None, :, :]
            action, _ = agent.select_action(state)
            pysc2_action = None  # TODO: Map action index to pysc2 FunctionCall
            obs = env.step(pysc2_action)
            ep_reward += obs.reward
            done = obs.last()
        total_rewards.append(ep_reward)
        print(f"Episode {ep}: Reward={ep_reward}")
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

# Example usage:
# with open('../config/beacon_config.yaml') as f:
#     cfg = yaml.safe_load(f)
# env = BeaconEnv()
# agent = A2CAgent((1,64,64), n_actions=16)
# evaluate(agent, env, num_episodes=5)