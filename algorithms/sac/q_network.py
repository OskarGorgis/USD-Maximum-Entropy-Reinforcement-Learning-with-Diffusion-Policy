from typing import Sequence
from flax import linen as nn
import jax.numpy as jnp

class QNetwork(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(1)(x)
