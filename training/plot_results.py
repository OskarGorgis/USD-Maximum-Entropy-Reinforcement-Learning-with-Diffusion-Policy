import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


def plot_results_maxentdp(envs=[], results_dir="/results", agent_name="maxentdp"):
    for env_name in envs:
        logs_dir = os.path.join(results_dir, env_name, "logs")
        plots_dir = os.path.join(results_dir, env_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        train_csv = os.path.join(logs_dir, f"train_{agent_name}.csv")
        if os.path.exists(train_csv):
            df = pd.read_csv(train_csv)
            steps = df["step"]

            # actor loss plot
            plt.figure()
            plt.plot(steps, df["actor_loss_mean"], label="Actor loss")
            plt.xlabel("Environment steps")
            plt.ylabel("Actor loss")
            plt.title(f"SAC actor loss - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "actor_loss_mean.png"))
            plt.close()

            # critic losses
            plt.figure()
            plt.plot(steps, df["c_loss_mean_1"], label="Critic 1 loss")
            plt.plot(steps, df["c_loss_mean_2"], label="Critic 2 loss")
            plt.xlabel("Environment steps")
            plt.ylabel("Critic loss (MSE)")
            plt.title(f"SAC critic losses - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "critic_losses.png"))
            plt.close()

            # entropy
            plt.figure()
            plt.plot(steps, df["log_prob"], label="Entropy")
            plt.xlabel("Environment steps")
            plt.ylabel("Entropy")
            plt.title(f"SAC entropy - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "entropy.png"))
            plt.close()
        else:
            print(f"[WARN] No train.csv for {env_name}")

        # episode return
        returns_csv = os.path.join(logs_dir, f"returns_{agent_name}.csv")
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
            plt.title(f"SAC episode return - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "episode_return.png"))
            plt.close()
        else:
            print(f"[WARN] No returns.csv for {env_name}")

        print(f"[OK] Plots generated for {env_name}")


def plot_results_sac(envs, results_dir="results"):
    for env_name in envs:
        logs_dir = os.path.join(results_dir, env_name, "logs")
        plots_dir = os.path.join(results_dir, env_name, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        train_csv = os.path.join(logs_dir, "train.csv")
        if os.path.exists(train_csv):
            df = pd.read_csv(train_csv)
            steps = df["step"]

            # actor loss
            plt.figure()
            plt.plot(steps, df["actor_loss"], label="Actor loss")
            plt.xlabel("Environment steps")
            plt.ylabel("Actor loss")
            plt.title(f"SAC actor loss - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "actor_loss.png"))
            plt.close()

            # critic losses
            plt.figure()
            plt.plot(steps, df["critic1_loss"], label="Critic 1 loss")
            plt.plot(steps, df["critic2_loss"], label="Critic 2 loss")
            plt.xlabel("Environment steps")
            plt.ylabel("Critic loss (MSE)")
            plt.title(f"SAC critic losses - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "critic_losses.png"))
            plt.close()

            # entropy
            plt.figure()
            plt.plot(steps, df["entropy"], label="Entropy")
            plt.xlabel("Environment steps")
            plt.ylabel("Entropy")
            plt.title(f"SAC entropy - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "entropy.png"))
            plt.close()

            # temperature
            plt.figure()
            plt.plot(steps, df["temperature"], label="Temperature (alpha)")
            plt.xlabel("Environment steps")
            plt.ylabel("Temperature (alpha)")
            plt.title(f"SAC temperature - {env_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "temperature.png"))
            plt.close()
        else:
            print(f"[WARN] No train.csv for {env_name}")

        # return from episodes
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
agent_name="sac"
):
    csv_path=f"results/{agent_name}/multigoal/logs/train.csv"
    out_dir=f"results/{agent_name}/multigoal/plots"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    steps = df["step"]

    # reward plot
    plt.figure()
    plt.plot(steps, df["reward"])
    plt.xlabel("Training steps")
    plt.ylabel("Reward")
    plt.title("Reward over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward.png"))
    plt.close()

    # entropy plot
    plt.figure()
    plt.plot(steps, df["entropy"])
    plt.xlabel("Training steps")
    plt.ylabel("Entropy")
    plt.title("Policy entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy.png"))
    plt.close()

    # temp plot
    plt.figure()
    plt.plot(steps, df["temperature"])
    plt.xlabel("Training steps")
    plt.ylabel("Temperature alpha")
    plt.title("Entropy temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temperature.png"))
    plt.close()

    # closest goal plot
    plt.figure()
    plt.step(steps, df["closest_goal"], where="post")
    plt.xlabel("Training steps")
    plt.ylabel("Closest goal index")
    plt.title("Active goal over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "closest_goal.png"))
    plt.close()

    # trajectory plot
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

def parse_eval_return(value):
    if isinstance(value, (int, float, np.number)):
        return float(value), 0.0

    if not isinstance(value, str):
        return np.nan, np.nan

    nums = re.findall(r"-?\d+\.\d+|-?\d+", value)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    elif len(nums) == 1:
        return float(nums[0]), 0.0
    else:
        return np.nan, np.nan


def compare_agents_all_metrics(
    env_name,
    results_dir="results",
    out_dir="results/comparisons",
    every_n=1,
    max_step=500_000,
):

    os.makedirs(out_dir, exist_ok=True)

    AGENTS = {
        "SAC": {
            "train": "train.csv",
            "returns": "returns.csv",
            "eval": "eval_returns.csv",
            "actor_loss": "actor_loss",
        },
        "MaxEntDP": {
            "train": "train_maxentdp.csv",
            "returns": "returns_maxentdp.csv",
            "eval": "eval_returns_maxentdp.csv",
            "actor_loss": "actor_loss_mean",
        },
    }

    # ============================================================
    # 1) EVALUATION RETURN (mean ± std)
    # ============================================================
    plt.figure(figsize=(7, 5))

    for label, spec in AGENTS.items():
        path = os.path.join(
            results_dir,
            label.lower(),
            env_name,
            "logs",
            spec["eval"],
        )

        if not os.path.exists(path):
            print(f"[WARN] Missing eval file for {label}")
            continue

        df = pd.read_csv(path)
        df = df[df["step"] <= max_step].iloc[::every_n]

        means, stds = [], []

        for v in df["eval_return"]:
            if isinstance(v, str):
                mean, std = parse_eval_return(v)
                means.append(mean)
                stds.append(std)
            else:
                means.append(v)
                stds.append(0.0)

        steps = df["step"].values
        means = pd.Series(means).astype(float)
        stds = pd.Series(stds).astype(float)

        plt.plot(steps, means, label=label, linewidth=2)
        if stds.max() > 0:
            plt.fill_between(steps, means - stds, means + stds, alpha=0.2)

    plt.xlabel("Training steps")
    plt.ylabel("Evaluation return")
    plt.title(f"Evaluation return – {env_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{env_name}_eval_return_compare.png"))
    plt.close()

    # ============================================================
    # 2) TRAINING EPISODE RETURN
    # ============================================================
    plt.figure(figsize=(7, 5))

    for label, spec in AGENTS.items():
        path = os.path.join(
            results_dir,
            label.lower(),
            env_name,
            "logs",
            spec["returns"],
        )

        if not os.path.exists(path):
            print(f"[WARN] Missing returns file for {label}")
            continue

        df = pd.read_csv(path)
        df = df[df["step"] <= max_step].iloc[::every_n]

        plt.plot(
            df["step"],
            df["episode_return"],
            label=label,
            linewidth=2,
        )

    plt.xlabel("Training steps")
    plt.ylabel("Episode return")
    plt.title(f"Training episode return – {env_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{env_name}_episode_return_compare.png"))
    plt.close()

    # ============================================================
    # 3) ACTOR LOSS (OPTIONAL, ALE JEST)
    # ============================================================
    plt.figure(figsize=(7, 5))
    plotted = False

    for label, spec in AGENTS.items():
        path = os.path.join(
            results_dir,
            label.lower(),
            env_name,
            "logs",
            spec["train"],
        )

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if spec["actor_loss"] not in df.columns:
            continue

        df = df[df["step"] <= max_step].iloc[::every_n]

        plt.plot(
            df["step"],
            df[spec["actor_loss"]],
            label=label,
            linewidth=2,
        )
        plotted = True

    if plotted:
        plt.xlabel("Training steps")
        plt.ylabel("Actor loss")
        plt.title(f"Actor loss – {env_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{env_name}_actor_loss_compare.png"))
    plt.close()

    print(f"[OK] Comparison plots generated for {env_name}")

def compare_agents_plots(envs):
    for env_name in envs:
        compare_agents_all_metrics(env_name)