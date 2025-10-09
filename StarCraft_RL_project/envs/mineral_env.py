
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import flags

class MineralEnv:
    def __init__(self, step_mul=8, screen_size=64, minimap_size=64, visualize=True):
        # Bắt buộc phải khởi tạo FLAGS để tránh lỗi khi chạy nhiều lần
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(['program'])

        # Tạo môi trường StarCraft II
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
        """Reset lại môi trường và trả về quan sát đầu tiên"""
        obs = self.env.reset()
        return obs[0]

    def step(self, actions_list):
        """Thực hiện một bước trong môi trường"""
        obs = self.env.step(actions_list)
        return obs[0]

    def close(self):
        """Đóng môi trường"""
        self.env.close()
