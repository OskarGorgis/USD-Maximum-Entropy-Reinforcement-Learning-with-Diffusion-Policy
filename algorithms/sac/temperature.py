from flax import linen as nn
import jax.numpy as jnp


class Temperature(nn.Module):
    init_temperature: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param(
            "log_alpha",
            lambda key: jnp.log(jnp.array(self.init_temperature)),
        )
        return jnp.exp(log_alpha)
