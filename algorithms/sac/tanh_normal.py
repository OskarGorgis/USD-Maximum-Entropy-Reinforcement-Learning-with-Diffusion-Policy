from flax import linen as nn
import jax
import jax.numpy as jnp


class TanhGaussian(nn.Module):
    base_cls: nn.Module
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations):
        x = self.base_cls()(observations)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return TanhGaussianDistribution(mean, log_std)

class TanhGaussianDistribution:
    def __init__(self, mean, log_std):
        self.mean = mean
        self.log_std = log_std
        self.std = jnp.exp(log_std)

    def sample(self, seed):
        eps = jax.random.normal(seed, self.mean.shape)
        z = self.mean + self.std * eps
        action = jnp.tanh(z)
        return action

    def log_prob(self, action):
        # inverse tanh
        eps = 1e-6
        z = 0.5 * jnp.log((1 + action + eps) / (1 - action + eps))

        log_prob = -0.5 * (
            ((z - self.mean) / self.std) ** 2
            + 2 * self.log_std
            + jnp.log(2 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)
        log_prob -= jnp.sum(jnp.log(1 - action**2 + eps), axis=-1)
        return log_prob
