from jax.nn import initializers as jinit
from flax import linen as nn
from .torch_layers import *
from jax import numpy as jnp
from typing import Callable, Any

standard_normal_init = jinit.normal(1)


class GromovMLP(nn.Module):
    activation: Callable
    n_classes: int
    depth: int
    width: int
    d_vocab: int
    use_bias: bool

    @nn.compact
    def __call__(self, x):
        for d in range(self.depth-1):
            x = jnp.sqrt(1 / x.shape[1]) * nn.Dense(self.width, use_bias=False, kernel_init=standard_normal_init)(x)
            x = x ** 2
        x = jnp.sqrt(1 / x.shape[1]) * nn.Dense(self.n_classes, use_bias=False, kernel_init=standard_normal_init)(x)
        return x


class MLP(nn.Module):
    activation: Callable
    n_classes: int
    depth: int
    width: int
    d_vocab: int
    use_bias: bool

    @nn.compact
    def __call__(self, x):
        # x = jnp.atleast_2d(x)
        # x = TorchEmbed(self.d_vocab, self.width)(x)        # x = x.reshape(x.shape[0], -1)
        for d in range(self.depth - 1):
            x = TorchLinear(self.width, use_bias=self.use_bias)(x)
            x = self.activation(x)
        x = TorchLinear(self.n_classes, use_bias=self.use_bias)(x)
        return x


class NormalizedMLP(nn.Module):
    activation: Callable
    n_classes: int
    width: int
    depth: int
    d_vocab: int
    normalization_scale: Any
    normalization_bias: Any
    use_bias: bool

    @nn.compact
    def __call__(self, x):
        # x = jnp.atleast_2d(x)
        # x = TorchEmbed(self.d_vocab, self.width)(x)
        # x = x.reshape(x.shape[0], -1)
        for d in range(self.depth-1):  # depth = 2
            x = TorchLinear(self.width, use_bias=self.use_bias)(x)
            x = TorchFixedBN(use_running_average=False)(x)
            x = self.activation(x)
        x = TorchLinear(self.n_classes, use_bias=self.use_bias)(x)
        x = self.normalization_scale * TorchFixedBN(use_running_average=False)(x)
        return x


class HalfNormalizedMLP(nn.Module):
    activation: Callable
    n_classes: int
    width: int
    depth: int
    use_bias: bool

    @nn.compact
    def __call__(self, x):
        # x = jnp.atleast_2d(x)
        # x = TorchEmbed(self.d_vocab, self.width)(x)
        # x = x.reshape(x.shape[0], -1)
        for d in range(self.depth-1):
            x = TorchLinear(self.width)(x)
            x = TorchFixedBN(use_running_average=False)(x)
            x = self.activation(x)
        x = TorchLinear(self.n_classes)(x)
        return x
