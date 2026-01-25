from algorithms.sac.sac import SACLearner
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

def evaluate_sac_policy(agent, env_name, num_episodes=5):
    env = MuJoCoEnv(env_name, seed=42)
    returns = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            dist = agent.actor.apply_fn(
                {"params": agent.actor.params},
                obs[None, :]
            )
            action = np.asarray(dist.mode())[0]
            obs, reward, done, _ = env.step(action)
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns))


def save_params(agent, step, env_name, prefix="sac/"):
    data = {
        "actor": to_bytes(agent.actor.params),
        "critic1": to_bytes(agent.critic1.params),
        "critic2": to_bytes(agent.critic2.params),
        "target_critic1": to_bytes(agent.target_critic1.params),
        "target_critic2": to_bytes(agent.target_critic2.params),
        "temp": to_bytes(agent.temp.params),
    }
    save_checkpoint(step, env_name, data, prefix=prefix)


def train_sac(env_name, total_steps=1_000_000, prefix = "sac/"):

    print("=== SAC Training ===")

    episode_return = 0.0
    episode_length = 0
    env = MuJoCoEnv(env_name, seed=0)

    agent = SACLearner.create(
        seed=0,
        observation_space=env.env.observation_space,
        action_space=env.env.action_space,
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
            dist = agent.actor.apply_fn(
                {"params": agent.actor.params},
                obs[None, :]
            )
            key, rng = jax.random.split(agent.rng)
            action, _ = dist.sample(seed=key)
            action = np.asarray(action[0])
            agent = agent.replace(rng=rng)

        next_obs, reward, done, _ = env.step(action)
        episode_return += reward
        episode_length += 1
        buffer.add(obs, action, reward, next_obs, done)

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
                log_to_file(step, info, env_name, prefix=prefix)

        if step % 50_000 == 0 and step > 0:
            save_params(agent, step, env_name, prefix=prefix)

        if step % 10_000 == 0 and step > 0:
            eval_return = evaluate_sac_policy(agent, env_name)
            log_eval_return(step, eval_return, env_name, prefix=prefix)

    env.close()
