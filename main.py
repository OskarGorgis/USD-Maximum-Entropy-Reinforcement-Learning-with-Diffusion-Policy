from enviroments.mujoco import ENVS
from training.train import train
from training.train_maxentdp import train_maxentdp
from training.plot_results import plot_results, plot_multigoal_results
from training.train_multigoal import train_multigoal


if __name__ == "__main__":
    # train_multigoal(
    #     total_steps=200_000,
    #     seed=0,
    # )
    plot_multigoal_results()
    # for env_name in ENVS:
    #     print(f"\n=== Training {env_name} ===")
    #     #train(env_name)
    #     train_maxentdp(env_name)
    # plot_results()
