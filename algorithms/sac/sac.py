from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from .temperature import Temperature
from .tanh_normal import TanhGaussian
from .mlp import MLP
from .q_network import QNetwork

@struct.dataclass
class SACLearner:
    rng: jax.random.PRNGKey

    actor: TrainState

    critic1: TrainState
    critic2: TrainState

    target_critic1: TrainState
    target_critic2: TrainState

    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
    ):


        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhGaussian(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )


        critic1_def = QNetwork(hidden_dims)
        critic2_def = QNetwork(hidden_dims)

        critic1_params = critic1_def.init(critic_key, observations, actions)["params"]
        critic2_params = critic2_def.init(critic_key, observations, actions)["params"]

        critic1 = TrainState.create(
            apply_fn=critic1_def.apply,
            params=critic1_params,
            tx=optax.adam(critic_lr),
        )

        critic2 = TrainState.create(
            apply_fn=critic2_def.apply,
            params=critic2_params,
            tx=optax.adam(critic_lr),
        )

        target_critic1_def = QNetwork(hidden_dims)
        target_critic2_def = QNetwork(hidden_dims)

        target_critic1_params = target_critic1_def.init(critic_key, observations, actions)["params"]
        target_critic2_params = target_critic2_def.init(critic_key, observations, actions)["params"]

        target_critic1 = TrainState.create(
            apply_fn=target_critic1_def.apply,
            params=target_critic1_params,
            tx=optax.adam(critic_lr),
        )

        target_critic2 = TrainState.create(
            apply_fn=target_critic2_def.apply,
            params=target_critic2_params,
            tx=optax.adam(critic_lr),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,

            critic1=critic1,
            critic2=critic2,

            target_critic1=target_critic1,
            target_critic2=target_critic2,

            temp=temp,

            tau=tau,
            discount=discount,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
        )

    def update_actor(self, batch):
        key, rng = jax.random.split(self.rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            q1 = self.critic1.apply_fn(
                {"params": self.critic1.params},
                batch["observations"],
                actions,
            )

            q2 = self.critic2.apply_fn(
                {"params": self.critic2.params},
                batch["observations"],
                actions,
            )

            q = jnp.minimum(q1, q2)

            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float):
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch):
        rng = self.rng

        # -------- sample next actions --------
        dist = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"],
        )

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # -------- target Q --------
        next_q1 = self.target_critic1.apply_fn(
            {"params": self.target_critic1.params},
            batch["next_observations"],
            next_actions,
        )

        next_q2 = self.target_critic2.apply_fn(
            {"params": self.target_critic2.params},
            batch["next_observations"],
            next_actions,
        )

        next_q = jnp.minimum(next_q1, next_q2)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                    self.discount
                    * batch["masks"]
                    * self.temp.apply_fn({"params": self.temp.params})
                    * next_log_probs
            )

        # -------- critic 1 update --------
        def critic1_loss_fn(params):
            q1 = self.critic1.apply_fn(
                {"params": params},
                batch["observations"],
                batch["actions"],
            )
            return ((q1 - target_q) ** 2).mean()

        grads1 = jax.grad(critic1_loss_fn)(self.critic1.params)
        critic1 = self.critic1.apply_gradients(grads=grads1)

        # -------- critic 2 update --------
        def critic2_loss_fn(params):
            q2 = self.critic2.apply_fn(
                {"params": params},
                batch["observations"],
                batch["actions"],
            )
            return ((q2 - target_q) ** 2).mean()

        grads2 = jax.grad(critic2_loss_fn)(self.critic2.params)
        critic2 = self.critic2.apply_gradients(grads=grads2)

        # -------- SOFT UPDATE TARGETS (TU JEST KLUCZ) --------
        target_critic1 = self.target_critic1.replace(
            params=optax.incremental_update(
                critic1.params,
                self.target_critic1.params,
                self.tau,
            )
        )

        target_critic2 = self.target_critic2.replace(
            params=optax.incremental_update(
                critic2.params,
                self.target_critic2.params,
                self.tau,
            )
        )

        return (
            self.replace(
                critic1=critic1,
                critic2=critic2,
                target_critic1=target_critic1,
                target_critic2=target_critic2,
                rng=rng,
            ),
            {
                "critic1_loss": critic1_loss_fn(critic1.params),
                "critic2_loss": critic2_loss_fn(critic2.params),
            },
        )

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch, utd_ratio=1):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}