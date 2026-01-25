from enviroments.mujoco import ENVS
from training.train_sac import train_sac
from training.train_maxentdp import train_maxentdp
from training.plot_results import plot_results_maxentdp, plot_results_sac, plot_multigoal_results, compare_agents_plots
from training.train_multigoal_sac import train_multigoal_sac
from training.train_multigoal_maxentdp import train_multigoal_maxentdp


if __name__ == "__main__":
    for env_name in ENVS:
        print(f"\n=== Training {env_name} ===")

        # there are 2 different training fucntions because
        # process of training i different for both algorithms (main difference diffusion generation)
        train_sac(env_name)         # sac training with logging
        train_maxentdp(env_name)    # maxentdp training with logging

        # there are 2 plotters because parameters of models are called different in files, it is possible to change with refacorization of whole algorithms code
        plot_results_sac(envs=ENVS)
        plot_results_maxentdp(envs=ENVS)

    # training both algorithms, also 2 diffeerent training processes
    train_multigoal_sac()
    train_multigoal_maxentdp()

    # plotting reults from multigoal - different plotters because there is 2D space to plot with trajectory
    plot_multigoal_results(agent="sac")
    plot_results_maxentdp(agent="maxentdp")

    compare_agents_plots(envs=ENVS)



