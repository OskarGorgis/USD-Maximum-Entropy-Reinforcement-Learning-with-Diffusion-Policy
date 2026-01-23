from algorithms.maxentdp.max_entropy_dp import MaxEntropyLearner
from enviroments.multigoal_point import MultiGoalPointEnv
from .replay_buffer import ReplayBuffer
from flax.serialization import to_bytes
import numpy as np
import jax
import csv
import os
from tqdm import tqdm
from .logger import save_checkpoint

def log_multigoal_maxentdp(
    step: int,
    obs,
    reward: float,
    closest_goal: int,
    info: dict,
    csv_path="results/MaxEntDP/multigoal/logs/train.csv"
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Pobieramy statystyki specyficzne dla MaxEntDP
    actor_loss = info.get("actor_loss_mean", 0.0)
    c1_loss = info.get("c_loss_mean_1", 0.0)
    c2_loss = info.get("c_loss_mean_2", 0.0)
    
    # Pozycja agenta (dla wizualizacji "kwiatka")
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
                "c2_loss"
            ])

        writer.writerow([
            step,
            x,
            y,
            closest_goal,
            reward,
            actor_loss,
            c1_loss,
            c2_loss
        ])

def save_params_maxentdp(agent, step, prefix="MaxEntDP/"):
    params_to_save = {
        "score_model": to_bytes(agent.score_model.params),
        "critic_1": to_bytes(agent.critic_1.params),
        "critic_2": to_bytes(agent.critic_2.params),
        "target_critic_1": to_bytes(agent.target_critic_1.params),
        "target_critic_2": to_bytes(agent.target_critic_2.params),
    }
    # Używamy save_checkpoint z Twojego loggera, dostosowując argumenty
    # Zakładam, że save_checkpoint przyjmuje (step, params, env_name, ...)
    # Jeśli sygnatura jest inna, trzeba to dostosować
    save_checkpoint(step, params_to_save, "Multigoal", prefix=prefix)

def train_multigoal_maxentdp(
    total_steps=200_000,
    seed=0,
    buffer_size=200_000,
    batch_size=256,
    prefix="MaxEntDP/"
):
    print("=== MaxEntDP Training (MultiGoal) ===")
    
    # --- ENV ---
    env = MultiGoalPointEnv()
    
    # --- AGENT ---
    # Dla prostego środowiska 2D temp=0.1 lub 0.2 powinno być ok na start
    # Możesz to dostroić, jeśli agent będzie zbyt chaotyczny
    agent = MaxEntropyLearner.create(
        seed=seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        temp=0.2, 
        # Zmniejszamy sieć dla prostego środowiska, żeby było szybciej
        # Domyślne (256, 256) też zadziałają, ale (128, 128) wystarczą
        actor_hidden_dims=(128, 128),
        critic_hidden_dims=(128, 128)
    )

    # --- REPLAY BUFFER ---
    buffer = ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        size=buffer_size,
    )

    obs, _ = env.reset()
    
    # Słownik na info z update'u
    info = {}

    for step in tqdm(range(total_steps)):

        # --- ACTION ---
        if step < 5_000: # Mniejszy warmup dla prostego env
            action = env.action_space.sample()
        else:
            # Używamy API MaxEntDP (sample_actions zwraca krotkę (akcja, nowy_agent))
            action, agent = agent.sample_actions(obs)
            action = np.asarray(action)

        # --- STEP ---
        next_obs, reward, done, _, _ = env.step(action)

        # --- KTÓRY CEL JEST NAJBLIŻEJ (do logowania) ---
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
            False,  # W MultiGoalPointEnv epizod nigdy się nie kończy "naturalnie"
        )

        obs = next_obs

        # --- UPDATE ---
        if step > 1_000:
            batch = buffer.sample(batch_size)
            agent, info = agent.update(batch)

        # --- LOGGING ---
        if step % 500 == 0 and step > 0:
            log_multigoal_maxentdp(
                step=step,
                obs=obs,
                reward=reward,
                closest_goal=closest_goal,
                info=info,
            )

        # --- SAVING ---
        if step % 50_000 == 0 and step > 0:
            save_params_maxentdp(agent, step, prefix=prefix)

    return agent