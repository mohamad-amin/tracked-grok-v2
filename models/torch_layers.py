import torch
from jax import random
from flax import linen as nn
from jax import numpy as jnp
from functools import partial

torch_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
torch_init_fanout = nn.initializers.variance_scaling(1 / 3, "fan_out", "uniform")
# This is not really torch init! This is fan_out, but we actually need fan_in. Can't see a clear way of doing this.
torch_bias_init = lambda k, s, d: (random.uniform(k, s, d) * (2 / jnp.sqrt(s[0]))) - (1 / jnp.sqrt(s[0]))

he_fan_in = nn.initializers.variance_scaling(1, "fan_out", "normal")
he_fan_out = nn.initializers.variance_scaling(1, "fan_out", "normal")

TorchLinear = partial(
    nn.Dense,
    kernel_init=torch_init,
    bias_init=torch_bias_init,
    # use_bias=False,
    dtype=None
)
TorchConv = partial(
    nn.Conv,
    kernel_init=torch_init,
    bias_init=torch_bias_init,
    dtype=None
)

TorchFixedBN = partial(
    nn.BatchNorm,
    momentum=0.0,
    epsilon=0,
    use_bias=False,
    use_scale=False,
    dtype=None
)


class TorchEmbed(nn.Module):
    d_vocab: int
    d_model: int

    def setup(self):
        self.W_E = self.param('W_E', he_fan_out, (self.d_model, self.d_vocab))

    def __call__(self, x):
        return jnp.einsum('dbp -> bpd', self.W_E[:, x])


class TorchUnembed(nn.Module):
    d_vocab: int
    d_model: int

    def setup(self):
        self.W_U = self.param('W_U', he_fan_in, (self.d_model, self.d_vocab))

    def __call__(self, x):
        return x @ self.W_U


class TorchPosEmbed(nn.Module):
    max_ctx: int
    d_model: int

    def setup(self):
        self.W_pos = self.param('W_pos', he_fan_in, (self.max_ctx, self.d_model))

    def __call__(self, x):
        return x + self.W_pos[:x.shape[-2]]
