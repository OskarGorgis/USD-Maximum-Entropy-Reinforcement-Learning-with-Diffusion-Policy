from typing import Dict

import gymnasium as gym
import numpy as np

# from jaxrl5.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    # if save_video:
    #     env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env.env, buffer_length=num_episodes)

    for ep_i in range(num_episodes):
        observation, info = env.reset()
        terminated = False
        trunacated = False
        i = 0
        while not terminated and not trunacated:
            # i+=1
            # if i % 100 == 0:
            #     print("step: ", i, "in episode: ", ep_i )
            action, agent = agent.eval_actions(observation)
            observation, reward, terminated, trunacated, info = env.step(action)
    return {"return": np.mean(env.return_queue), "return_std": np.std(env.return_queue)}

def implicit_evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    # if save_video:
    #     env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.sample_implicit_policy(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}
