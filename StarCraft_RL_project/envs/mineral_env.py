import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import flags

class MineralEnv:
    def __init__(self, step_mul=8, screen_size=64, minimap_size=64, visualize=True):
        flags.FLAGS(['program'])
        self.env = sc2_env.SC2Env(
            map_name='CollectMineralShards',
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=screen_size, minimap=minimap_size),
                use_feature_units=True
            ),
            step_mul=step_mul,
            visualize=visualize
        )

    def reset(self):
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        obs = self.env.step([action])
        return obs[0]

    def close(self):
        self.env.close()
