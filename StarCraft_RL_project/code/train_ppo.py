import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from sc2_gym_wrapper import MoveToBeaconWrapper

# === Callback để log và lưu checkpoint ===
class TrainAndSaveCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(TrainAndSaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # cứ mỗi check_freq steps thì log và save
        if self.num_timesteps % self.check_freq == 0:
            mean_reward = self.training_env.get_attr("episode_rewards")[-100:] \
                          if hasattr(self.training_env.envs[0], "episode_rewards") else None
            if mean_reward:
                avg = sum(mean_reward) / len(mean_reward)
                print(f"Step: {self.num_timesteps} | Mean Reward (last 100 eps): {avg:.2f}")

            model_path = os.path.join(self.save_path, f"ppo_sc2_{self.num_timesteps}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"✅ Saved checkpoint to {model_path}")
        return True


def main():
    # Khởi tạo env
    env = DummyVecEnv([lambda: MoveToBeaconWrapper()])
    env = VecTransposeImage(env)

    # PPO config (có thể chỉnh để ổn định hơn)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,   # giảm để tránh KL quá cao
        n_steps=4096,         # tăng số bước trước khi update
        batch_size=256,       # batch lớn hơn, học ổn định hơn
    )

    # callback log mỗi 10k steps
    callback = TrainAndSaveCallback(check_freq=10000, save_path="./checkpoints/")

    # Train
    model.learn(total_timesteps=200000, callback=callback)  # train ít nhất 200k bước

    # Lưu model cuối
    model.save("ppo_move_to_beacon_final")
    print("🎉 Training xong, model đã lưu vào ppo_move_to_beacon_final.zip")


if __name__ == "__main__":
    main()
