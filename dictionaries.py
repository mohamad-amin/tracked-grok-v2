import jax
from flax import linen as nn
from jax import numpy as jnp
from models.simple import MLP, NormalizedMLP, HalfNormalizedMLP, GromovMLP
# from models.resnet import ResNet18
# from models.vgg import vgg11
# from models.bert import bert_tiny
# from models.transformer import Transformer
from functools import partial
from data import (
    modular_addition,
    mnist,
    cifar10,
    cifar10_binary,
    sst2,
)

data_dict = {
    "cifar10": cifar10,
    "mnist": mnist,
    "cifar10_binary": cifar10_binary,
    "sst2": sst2,
    "modular_addition": modular_addition,
}

model_dict = {
    "mlp": MLP,
    "gromov_mlp": GromovMLP,
    # "resnet18": ResNet18,
    # "vgg": vgg11,
    # "bert_tiny": bert_tiny,
    # "transformer": Transformer,
    "normalized_mlp": NormalizedMLP,
    "half_normalized_mlp": HalfNormalizedMLP,
}

CrossEntropyLoss = lambda f, y: -nn.log_softmax(f)[y]
CategoricalMSELoss = lambda f, y: (f @ f - 2 * f[y] + 1) / len(f)
CategoricalL1Loss = lambda f, y: (jnp.abs(f).sum() - jnp.abs(f[y]) + jnp.abs(f[y] - 1)) / len(f)
BinaryMSELoss = lambda f, y: jnp.mean((f[0] - y) ** 2)
LogisticLoss = lambda f, y: -nn.log_sigmoid(f[0] * y)
criterion_dict = {
    "cat_mse": CategoricalMSELoss,
    "cat_l1": CategoricalL1Loss,
    "binary_mse": BinaryMSELoss,
    "ce": CrossEntropyLoss,
    "logistic": LogisticLoss,
}


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)


activation_dict = {
    "eye": lambda x: x,
    "quadratic": lambda x: x**2,
    "cubic": lambda x: x**3,
    "quartic": lambda x: x**4,
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}
