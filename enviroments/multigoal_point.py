import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.special import logsumexp

class MultiGoalPointEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.dt = 0.05

        # definition of goals
        self.goals = np.array([
            [ 1.0,  1.0],
            [ 1.0, -1.0],
            [-1.0,  1.0],
            [-1.0, -1.0],
        ])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,)
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros(2)
        return self.pos.copy(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.pos = self.pos + self.dt * action

        # distances from goals
        distances = np.linalg.norm(self.goals - self.pos[None, :], axis=1)


        beta = 5.0  # higher - harder problem
        reward = np.log(np.sum(np.exp(-beta * distances))) / beta

        done = False

        return self.pos.copy(), reward, done, False, {}
