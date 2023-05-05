from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import numpy as np
from jax import random
from jax import numpy as jnp
import os
from functools import partial


def modular_addition(data_config, width):

    def get_fn(fn_name, p):
        random_answers = np.random.randint(low=0, high=p, size=(p, p))
        return {
            'add': lambda x, y: (x + y) % p,
            'subtract': lambda x, y: (x - y) % p,
            'x2xyy2': lambda x, y: (x ** 2 + x * y + y ** 2) % p,
            'rand': lambda x, y: random_answers[x][y]
        }[fn_name]

    def _to_quadratic(X):
        return jnp.einsum('ij,ik->ijk', X, X).reshape(X.shape[0], -1)

    p = data_config['p']
    seed = data_config['seed']
    fn_name = data_config['fn_name']
    frac_train = data_config['frac_train']
    embed_dim = data_config.get('embed_dim', width)  # Dimension of each symbol after embedding
    lookup_type = data_config.get('lookup_type', 'gaussian')
    quadratic_data = data_config.get('is_quadratic', False)
    assert lookup_type in ['gaussian', 'one_hot'], "Lookup type should either be 'gaussian' or 'one_hot'"
    fn = get_fn(fn_name, p)

    # Generate train and test split
    pairs = jnp.array([(i, j) for i in range(p) for j in range(p)])
    key = random.PRNGKey(seed)
    pairs = random.permutation(key, pairs, axis=0)
    div = int(frac_train * len(pairs))

    x_train = pairs[:div]
    y_train = jnp.array([fn(i, j) for i, j in x_train])
    x_test = pairs[div:]
    y_test = jnp.array([fn(i, j) for i, j in x_test])

    key = random.PRNGKey(seed)
    if lookup_type == 'gaussian':
        lookup_table = random.normal(key, (embed_dim, p), dtype=jnp.float32) / jnp.sqrt(width)
    elif lookup_type == 'one_hot':
        lookup_table = jnp.zeros((embed_dim, p), dtype=jnp.float32)
        perm = random.permutation(key, jnp.arange(p), independent=True)
        lookup_table = lookup_table.at[perm, jnp.arange(p)].set(1)
    map_to_lookup = lambda data: jnp.stack([lookup_table[:, x].reshape(-1, ) for x in data])

    x_train = map_to_lookup(x_train)
    x_test = map_to_lookup(x_test)

    if quadratic_data:
        x_train = _to_quadratic(x_train)
        x_test = _to_quadratic(x_test)

    return (x_train, y_train), (x_test, y_test)


def cifar10(data_config):

    n = data_config['train_count']

    # trainset = load_dataset("cifar10")["train"]
    trainset = load_from_disk("data_files")["train"]
    x = np.stack(trainset[:n]["img"])
    y = np.array(trainset[:n]["label"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    x = (x - x.mean((0, 1, 2))) / x.std((0, 1, 2))

    return x, y


def mnist(data_config):

    n = data_config['train_count']

    trainset = load_from_disk("mnist_files")["train"]
    x = np.stack(trainset[:n]["image"])
    y = np.array(trainset[:n]["label"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    x = (x - x.mean((0, 1, 2))) / x.std((0, 1, 2))
    return x, y


def cifar10_binary(data_config):

    n = data_config['train_count']

    # trainset = load_dataset("cifar10")["train"]
    trainset = load_from_disk("data_files")["train"]
    y = np.array(trainset["label"])
    cat_idx = np.argwhere(y == 3)[: n // 2, 0]
    dog_idx = np.argwhere(y == 5)[: n // 2, 0]
    idx = np.concatenate([cat_idx, dog_idx])
    y = 2 * (y[idx] == 5) - 1
    x = np.stack(trainset[idx]["img"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    x = (x - x.mean((0, 1, 2))) / x.std((0, 1, 2))
    return x, y


def sst2(data_config):

    n = data_config['train_count']

    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "sst2")
    y = np.array(dataset["train"]["label"])
    neg_idx = np.argwhere(y == 0)[: n // 2, 0]
    pos_idx = np.argwhere(y == 1)[: n // 2, 0]
    idx = np.concatenate([neg_idx, pos_idx])
    y = 2 * y[idx] - 1
    x = dataset["train"][idx]["sentence"]
    x = tokenizer(x, padding=True)["input_ids"]
    x, y = jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)
    return x, y
