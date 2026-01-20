
def evaluate(agent, env, episodes=10):
    returns = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            dist = agent.actor.apply_fn(
                {"params": agent.actor.params},
                obs[None, :]
            )
            action = np.asarray(dist.mean())[0]  # BEZ LOSOWOŚCI
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    return np.mean(returns), np.std(returns)
