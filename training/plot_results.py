import os
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ENVS = [
    "Humanoid-v4",
    "Ant-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
]


def plot_results(envs=DEFAULT_ENVS, results_dir="results"):
    for env_name in envs:
        logs_dir = os.path.join(results_dir, env_name, "logs")
        plots_dir = os.path.join(results_dir, env_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # ================= TRAIN METRICS =================
        train_csv = os.path.join(logs_dir, "train.csv")
        if os.path.exists(train_csv):
            df = pd.read_csv(train_csv)
            steps = df["step"]

            # Actor loss
            plt.figure()
            plt.plot(steps, df["actor_loss"], label="Actor loss")
            plt.xlabel("Environment steps")
            plt.ylabel("Actor loss")
            plt.title(f"SAC actor loss – {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "actor_loss.png"))
            plt.close()

            # Critic losses
            plt.figure()
            plt.plot(steps, df["critic1_loss"], label="Critic 1 loss")
            plt.plot(steps, df["critic2_loss"], label="Critic 2 loss")
            plt.xlabel("Environment steps")
            plt.ylabel("Critic loss (MSE)")
            plt.title(f"SAC critic losses – {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "critic_losses.png"))
            plt.close()

            # Entropy
            plt.figure()
            plt.plot(steps, df["entropy"], label="Entropy")
            plt.xlabel("Environment steps")
            plt.ylabel("Entropy")
            plt.title(f"SAC entropy – {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "entropy.png"))
            plt.close()

            # Temperature
            plt.figure()
            plt.plot(steps, df["temperature"], label="Temperature (α)")
            plt.xlabel("Environment steps")
            plt.ylabel("Temperature (α)")
            plt.title(f"SAC temperature – {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "temperature.png"))
            plt.close()
        else:
            print(f"[WARN] No train.csv for {env_name}")

        # ================= EPISODE RETURN =================
        returns_csv = os.path.join(logs_dir, "returns.csv")
        if os.path.exists(returns_csv):
            df_ret = pd.read_csv(returns_csv)

            window = 10
            rolling_return = df_ret["episode_return"].rolling(window).mean()

            plt.figure()
            plt.plot(
                df_ret["step"],
                rolling_return,
                label=f"Episode return (rolling mean {window})",
            )
            plt.xlabel("Environment steps")
            plt.ylabel("Episode return")
            plt.title(f"SAC episode return – {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "episode_return.png"))
            plt.close()
        else:
            print(f"[WARN] No returns.csv for {env_name}")

        print(f"[OK] Plots generated for {env_name}")


def plot_multigoal_results(
    csv_path="results/multigoal/logs/train.csv",
    out_dir="results/multigoal/plots",
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    steps = df["step"]

    # -------- Reward --------
    plt.figure()
    plt.plot(steps, df["reward"])
    plt.xlabel("Training steps")
    plt.ylabel("Reward")
    plt.title("Reward over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward.png"))
    plt.close()

    # -------- Entropy --------
    plt.figure()
    plt.plot(steps, df["entropy"])
    plt.xlabel("Training steps")
    plt.ylabel("Entropy")
    plt.title("Policy entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy.png"))
    plt.close()

    # -------- Temperature --------
    plt.figure()
    plt.plot(steps, df["temperature"])
    plt.xlabel("Training steps")
    plt.ylabel("Temperature α")
    plt.title("Entropy temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temperature.png"))
    plt.close()

    # -------- Closest goal (mode switching) --------
    plt.figure()
    plt.step(steps, df["closest_goal"], where="post")
    plt.xlabel("Training steps")
    plt.ylabel("Closest goal index")
    plt.title("Active goal over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "closest_goal.png"))
    plt.close()

    # -------- 2D trajectory --------
    plt.figure()
    plt.plot(df["x"], df["y"], linewidth=1.0)
    plt.scatter(
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        c="red",
        marker="x",
        label="Goals",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Agent trajectory in multi-goal space")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory.png"))
    plt.close()

    print(f"[OK] Multi-goal plots saved to {out_dir}")