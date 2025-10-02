from sc2_gym_wrapper import MoveToBeaconWrapper  # đổi tên file wrapper bạn đã lưu
import numpy as np

if __name__ == "__main__":
    env = MoveToBeaconWrapper()

    for ep in range(5):
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()  # random chọn 1 pixel
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {ep+1} finished after {steps} steps. Total reward = {total_reward}")

    env.close()
