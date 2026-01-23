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


def log_step_to_csv(
    step,
    episode_return,
    success_rate,
    entropy,
    temperature,
    unique_goals_ratio,
    csv_path="results/multigoal/logs/train.csv",
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # nagłówek tylko raz
        if not file_exists:
            writer.writerow([
                "step",
                "episode_return",
                "success_rate",
                "entropy",
                "temperature",
                "unique_goals_ratio",
            ])

        writer.writerow([
            step,
            episode_return,
            success_rate,
            entropy,
            temperature,
            unique_goals_ratio,
        ])


def train_multigoal(
    total_steps=200_000,
    seed=0,
    buffer_size=200_000,
    batch_size=256,
):
    # --- ENV ---
    env = MultiGoalPointEnv()

    # --- AGENT ---
    agent = SACLearner.create(
        seed=seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # --- REPLAY BUFFER ---
    buffer = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=buffer_size,
    )

    obs, _ = env.reset()

    for step in tqdm(range(total_steps)):

        # --- ACTION ---
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

        # --- STEP ---
        next_obs, reward, done, _, _ = env.step(action)

        # --- which goal is closest ---
        distances = np.linalg.norm(
            env.goals - next_obs[None, :],
            axis=1
        )
        closest_goal = int(np.argmin(distances))

        # --- BUFFER ---
        buffer.add(
            obs,
            action,
            reward,
            next_obs,
            False,  # NIGDY nie kończymy epizodu
        )

        obs = next_obs

        # --- UPDATE ---
        if step > 1_000:
            batch = buffer.sample(batch_size)
            agent, info = agent.update(batch)
        else:
            info = {}

        # --- LOGGING ---
        if step % 500 == 0 and step > 0:
            log_multigoal(
                step=step,
                obs=obs,
                reward=reward,
                closest_goal=closest_goal,
                info=info,
            )

    return agent



def save_checkpoint(agent, step):
    os.makedirs("results/multigoal/checkpoints", exist_ok=True)
    path = f"results/multigoal/checkpoints/sac_step_{step}.pkl"

    data = {
        "actor": to_bytes(agent.actor.params),
        "critic1": to_bytes(agent.critic1.params),
        "critic2": to_bytes(agent.critic2.params),
        "temp": to_bytes(agent.temp.params),
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)


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
