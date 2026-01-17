# environments/mujoco_env.py
import gymnasium as gym
import numpy as np

ENVS = [
    "Humanoid-v4",
    "Ant-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
]

class MuJoCoEnv:
    def __init__(self, env_name: str, seed: int = 0):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Seed
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        # Dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # Action bounds
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

    def reset(self):
        obs, _ = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.act_low, self.act_high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done, info

    def close(self):
        self.env.close()
