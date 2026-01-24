from algorithms.maxentdp.max_entropy_dp import MaxEntropyLearner
from enviroments.mujoco import MuJoCoEnv
from .replay_buffer import ReplayBuffer
from flax.serialization import to_bytes
import numpy as np
import jax
import pickle
import csv
import os
from tqdm import tqdm
from .logger import log_episode_return, log_eval_return, log_to_file, save_checkpoint

from jaxrl5.evaluation import evaluate


def save_params(agent, step, env_name, params_to_save, prefix="MaxEntDP/"):
    params_to_save = {
        "score_model": to_bytes(agent.score_model.params),
        "critic_1": to_bytes(agent.critic_1.params),
        "critic_2": to_bytes(agent.critic_2.params),
        "target_critic_1": to_bytes(agent.target_critic_1.params),
        "target_critic_2": to_bytes(agent.target_critic_2.params),
    }
    save_checkpoint(step, env_name, params_to_save, prefix=prefix)

def train_maxentdp(env_name, total_steps=500_000, prefix="MaxEntDP/"):

    print("=== MaxEntDP Training ===")

    temp = {
        "Hopper-v4" : 0.05 ,
        "Humanoid-v4" : 0.02,
        "Ant-v4" : 0.05,
        "Walker2d-v4" : 0.01,
        "HalfCheetah-v4" : 0.2,
    }

    episode_return = 0.0
    episode_length = 0
    env = MuJoCoEnv(env_name, seed=0)

    eval_env = MuJoCoEnv(env_name, seed=42)

    agent = MaxEntropyLearner.create(
        seed=0,
        observation_space=env.env.observation_space,
        action_space=env.env.action_space,
        temp=temp[env_name]
    )

    buffer = ReplayBuffer(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        size=1_000_000,
    )

    obs = env.reset()

    for step in tqdm(range(total_steps)):

        if step < 10_000:
            action = env.env.action_space.sample()
        else:
            action, agent = agent.sample_actions(obs)
            action = np.asarray(action)

        next_obs, reward, terminated, trunaceted, info = env.env.step(action)
        done = terminated or trunaceted
        # next_obs = np.array(next_obs)
        episode_return += reward
        episode_length += 1

     
        buffer.add(obs, action, reward, next_obs, terminated)

        if done:
            log_episode_return(
                step=step,
                episode_return=episode_return,
                episode_length=episode_length,
                env_name=env_name,
                prefix=prefix
            )
            episode_return = 0.0
            episode_length = 0
            obs = env.reset()
        else:
            obs = next_obs

        if step >= 1_000:
            batch = buffer.sample(256)
            agent, info = agent.update(batch)

            if step % 1_000 == 0:
                # print(info)
                log_to_file(step, info, env_name, prefix=prefix)

        if step % 50_000 == 0 and step > 0:
            save_params(agent, step, env_name, prefix=prefix)

        if step % 10_000 == 0 and step > 0:
            eval_return = evaluate(
                agent,
                eval_env,
                num_episodes=10,
                save_video=False,
            )
            log_eval_return(step, eval_return, env_name, prefix=prefix)

    env.close()
