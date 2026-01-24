from algorithms.sac.sac import SACLearner
from enviroments.multigoal_point import MultiGoalPointEnv
from .replay_buffer import ReplayBuffer
import numpy as np
import jax
import os
import csv
from flax.serialization import to_bytes
import pickle
from tqdm import tqdm

from .logger import save_checkpoint

def train_multigoal(
    total_steps=200_000,
    seed=0,
    buffer_size=200_000,
    batch_size=256,
):
    # env creation
    env = MultiGoalPointEnv()

    # agent creation
    agent = SACLearner.create(
        seed=seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # buffer
    buffer = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=buffer_size,
    )

    obs, _ = env.reset()

    for step in tqdm(range(total_steps)):

        # sampling actions
        if step < 1_000:
            action = env.action_space.sample()
        else:
            dist = agent.actor.apply_fn(
                {"params": agent.actor.params},
                obs[None]
            )
            key, rng = jax.random.split(agent.rng)
            action, logp = dist.sample(seed=key)
            action = np.asarray(action[0])
            agent = agent.replace(rng=rng)

        # next step in environment
        next_obs, reward, done, _, _ = env.step(action)

        # measure nearest goal for the logging
        distances = np.linalg.norm(
            env.goals - next_obs[None, :],
            axis=1
        )
        closest_goal = int(np.argmin(distances))

        # replay buffer update
        buffer.add(
            obs,
            action,
            reward,
            next_obs,
            False,
        )

        obs = next_obs

        # update agent with example batch after data collection period
        if step > 1_000:
            batch = buffer.sample(batch_size)
            agent, info = agent.update(batch)
        else:
            info = {}

        # log to file
        if step % 500 == 0 and step > 0:
            log_multigoal(
                step=step,
                obs=obs,
                reward=reward,
                closest_goal=closest_goal,
                info=info,
            )

    return agent


def log_multigoal(
    step: int,
    obs,
    reward: float,
    closest_goal: int,
    info: dict,
):


    os.makedirs("results/multigoal/logs", exist_ok=True)
    path = "results/multigoal/logs/train.csv"

    # bezpieczne pobranie statystyk
    entropy = info.get("entropy", 0.0)
    temperature = info.get("temperature", 0.0)

    # pozycja agenta
    x, y = float(obs[0]), float(obs[1])

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "step",
                "x",
                "y",
                "closest_goal",
                "reward",
                "entropy",
                "temperature",
            ])

        writer.writerow([
            step,
            x,
            y,
            closest_goal,
            reward,
            entropy,
            temperature,
        ])
