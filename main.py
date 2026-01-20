from enviroments.mujoco import ENVS
from training.train import train
from training.plot_results import plot_results


if __name__ == "__main__":
    for env_name in ENVS:
        print(f"\n=== Training {env_name} ===")
        train(env_name)
    plot_results()