import os, csv


def log_eval_return(step, eval_return, env_name, prefix=""):
    os.makedirs(f"results/{prefix}{env_name}/logs", exist_ok=True)
    path = f"results/{prefix}{env_name}/logs/eval_returns.csv"

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "eval_return"])
        writer.writerow([step, eval_return])

def log_episode_return(step, episode_return, episode_length, env_name, prefix=""):
    os.makedirs(f"results/{prefix}{env_name}/logs", exist_ok=True)
    path = f"results/{prefix}{env_name}/logs/returns.csv"

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "episode_return", "episode_length"])
        writer.writerow([step, episode_return, episode_length])

def log_to_file(step, info, env_name, prefix=""):
    os.makedirs(f"results/{prefix}{env_name}/logs", exist_ok=True)
    path = f"results/{prefix}{env_name}/logs/train_maxentdp.csv"
    # print(info)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["step"] + list(info.keys())
            )
        writer.writerow([step] + list(info.values()))