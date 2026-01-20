from flax import linen as nn
from typing import Sequence


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.hidden_dims):
            x = nn.Dense(h)(x)
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                x = nn.relu(x)
        return x
