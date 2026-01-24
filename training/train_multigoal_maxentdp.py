from algorithms.maxentdp.max_entropy_dp import MaxEntropyLearner
from enviroments.multigoal_point import MultiGoalPointEnv
from .replay_buffer import ReplayBuffer
from flax.serialization import to_bytes
import numpy as np
import jax
import csv
import os
from tqdm import tqdm

def train_multigoal_maxentdp(
    total_steps=200_000,
    seed=0,
    buffer_size=200_000,
    batch_size=256,
    prefix="maxentdp/"
):
    print("=== maxentdp Training (MultiGoal) ===")
    
    # env init
    env = MultiGoalPointEnv()
    
    # agent init
    agent = MaxEntropyLearner.create(
        seed=seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        temp=0.2, 
        # smaller networks - simpler task
        actor_hidden_dims=(128, 128),
        critic_hidden_dims=(128, 128)
    )

    # replay buffer init
    buffer = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=buffer_size,
    )

    obs, _ = env.reset()
    
    # Słownik na info z update'u
    info = {}

    for step in tqdm(range(total_steps)):

        # choosing action and taking samples from env
        if step < 5_000:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(obs)
            action = np.asarray(action)

        # next step in env
        next_obs, reward, done, _, _ = env.step(action)

        # distance from nearest goal for logging
        distances = np.linalg.norm(
            env.goals - next_obs[None, :],
            axis=1
        )
        closest_goal = int(np.argmin(distances))

        buffer.add(
            obs,
            action,
            reward,
            next_obs,
            False, 
        )

        obs = next_obs

        # take sample actions from buffer and update agent
        if step > 1_000:
            batch = buffer.sample(batch_size)
            agent, info = agent.update(batch)

        # log results to file
        if step % 500 == 0 and step > 0:
            log_multigoal_maxentdp(
                step=step,
                obs=obs,
                reward=reward,
                closest_goal=closest_goal,
                info=info,
            )

    return agent


def log_multigoal_maxentdp(
    step: int,
    obs,
    reward: float,
    closest_goal: int,
    info: dict,
    prefix: str="maxentdp"
    
):
    csv_path=f"results/{prefix}/multigoal/logs/train.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    actor_loss = info.get("actor_loss_mean", 0.0)
    c1_loss = info.get("c_loss_mean_1", 0.0)
    c2_loss = info.get("c_loss_mean_2", 0.0)
    entropy = info.get("log_prob", 0.0)
    temp = info.get("temp", 0.0)
    
    # agent position
    x, y = float(obs[0]), float(obs[1])

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "step",
                "x",
                "y",
                "closest_goal",
                "reward",
                "actor_loss",
                "c1_loss",
                "c2_loss",
                "entropy",
                "tempterature"
            ])

        writer.writerow([
            step,
            x,
            y,
            closest_goal,
            reward,
            actor_loss,
            c1_loss,
            c2_loss,
            entropy,
            temp
        ])
