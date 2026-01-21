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

def log_eval_return(step, eval_return, env_name):
    os.makedirs(f"results/{env_name}/logs", exist_ok=True)
    path = f"results/{env_name}/logs/eval_returns.csv"

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "eval_return"])
        writer.writerow([step, eval_return])

def log_episode_return(step, episode_return, episode_length, env_name):
    os.makedirs(f"results/{env_name}/logs", exist_ok=True)
    path = f"results/{env_name}/logs/returns.csv"

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "episode_return", "episode_length"])
        writer.writerow([step, episode_return, episode_length])

def evaluate_policy(agent, env_name, num_episodes=5):
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

def log_to_file(step, info, env_name):
    os.makedirs(f"results/{env_name}/logs", exist_ok=True)
    path = f"results/{env_name}/logs/train.csv"

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["step", "actor_loss", "critic1_loss", "critic2_loss", "entropy", "temperature"]
            )
        writer.writerow([
            step,
            info.get("actor_loss"),
            info.get("critic1_loss"),
            info.get("critic2_loss"),
            info.get("entropy"),
            info.get("temperature"),
        ])


def save_checkpoint(agent, step, env_name):
    os.makedirs(f"results/{env_name}/checkpoints", exist_ok=True)
    path = f"results/{env_name}/checkpoints/sac_step_{step}.npz"

    data = {
        "actor": to_bytes(agent.actor.params),
        "critic1": to_bytes(agent.critic1.params),
        "critic2": to_bytes(agent.critic2.params),
        "target_critic1": to_bytes(agent.target_critic1.params),
        "target_critic2": to_bytes(agent.target_critic2.params),
        "temp": to_bytes(agent.temp.params),
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)


def train(env_name, total_steps=1_000_000):
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
                log_to_file(step, info, env_name)

        if step % 50_000 == 0 and step > 0:
            save_checkpoint(agent, step, env_name)

        if step % 10_000 == 0 and step > 0:
            eval_return = evaluate_policy(agent, env_name)
            log_eval_return(step, eval_return, env_name)

    env.close()
