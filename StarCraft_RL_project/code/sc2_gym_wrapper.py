import gym
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import flags

# Fix lỗi FLAGS của absl
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(["sc2_gym_wrapper"])

class MoveToBeaconWrapper(gym.Env):
    """
    Gym wrapper cho PySC2 MoveToBeacon.
    Observation: ảnh 64x64 (screen layer player_relative).
    Action: chọn 1 pixel (0..4095) trên screen để Move_screen tới.
    """

    def __init__(self):
        super(MoveToBeaconWrapper, self).__init__()

        self.env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=64, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=False
        )

        # Observation: 64x64 grayscale
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
        )

        # Action: chọn pixel (0..4095)
        self.action_space = gym.spaces.Discrete(64 * 64)

        self.last_obs = None

    def _process_obs(self, obs):
        """Chuyển screen feature thành ảnh (64x64,1)."""
        screen = obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index]
        return np.expand_dims(screen, axis=-1).astype(np.uint8)

    def reset(self):
        obs = self.env.reset()[0]
        self.last_obs = obs
        return self._process_obs(obs)

    def step(self, action):
        x, y = action % 64, action // 64
        funcs = self.last_obs.observation["available_actions"]

        # Nếu đã có thể move -> move tới (x,y)
        if actions.FUNCTIONS.Move_screen.id in funcs:
            func = actions.FUNCTIONS.Move_screen("now", [x, y])
        # Nếu chưa select army -> select marine
        elif actions.FUNCTIONS.select_army.id in funcs:
            func = actions.FUNCTIONS.select_army("select")
        else:
            func = actions.FUNCTIONS.no_op()

        obs = self.env.step([func])[0]
        self.last_obs = obs

        reward = obs.reward
        done = obs.last()

        return self._process_obs(obs), reward, done, {}

    def close(self):
        self.env.close()
