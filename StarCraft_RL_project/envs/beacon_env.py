from pysc2.env import sc2_env
from pysc2.lib import actions

class BeaconEnv:
    def __init__(self):
        self.env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
                use_feature_units=True),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True
        )

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs = self.env.step([action])[0]
        return obs

    def close(self):
        self.env.close()
