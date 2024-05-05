import tqdm
import random
import itertools
from dataclasses import dataclass
from itertools import product
from typing import List, Any, Iterator
import numpy as np
from collections import deque

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random
from jax import Array
from jax.typing import ArrayLike

from grox.network import SimpleMLP, EmbeddingMatrix, layer
from grox.nn import softmax


@layer
class MLPModel(Layer):
    embedding: EmbeddingMatrix
    ffn_layers: mlp

    @classmethod
    def initialize(cls, key, num_tokens, dim_list, act_fn):
        assert dim_list[0] % 2 == 0

        key, embed_layer = EmbeddingMatrix.initialize(key, num_tokens, dim_list[0] // 2)
        key, mlp = MLPModel.initialize(key, dim_list, act_fn)
        return key, cls(embed_layer, mlp)

    def __call__(self, data):
        left_embed = self.embedding(data[:, 0])
        right_embed = self.embedding(data[:, 1])
        embeded = jnp.concatenate([left_embed, right_embed], axis=1)
        return self.ffn_layers(embeded)

    def tree_flatten(self):
        return ((self.embedding, self.ffn_layers), None)


def batch_iterator(array: ArrayLike, batchsize: int) -> Iterator[ArrayLike]:
    total = len(array)
    steps = total // batchsize
    actual_total = steps * batchsize

    for idx in range(0, actual_total, batchsize):
        data = array[idx : idx + batchsize]
        inputs = data[:, :2]
        targets = data[:, 3]
        yield (inputs, targets)


def nll_loss(probs: ArrayLike, target: ArrayLike) -> Array:
    log_probs = jnp.log(probs)
    ll = jnp.take_along_axis(log_probs, target.reshape(-1, 1), axis=1).mean()
    return -1.0 * ll


def compute_loss(mlp: MLPModel, data: ArrayLike, target: ArrayLike) -> Array:
    probs = softmax(mlp(data))
    loss_value = nll_loss(probs, target)
    return loss_value


def compute_acc(mlp: MLPModel, data: ArrayLike, target: ArrayLike) -> Array:
    logits = mlp(data)
    probs = softmax(logits)
    preds = jnp.argmax(probs, axis=1)
    return preds == target


def create_dataset(train_percent: float, max_value: int, fn) -> ArrayLike:
    nums = list(range(0, max_value))
    combos = list(itertools.product(nums, nums))
    data = list(map(lambda x: (*x, fn(*x)), combos))

    random.shuffle(data)

    total = len(data)
    train_amount = int(train_percent * total)

    train_data = jnp.array(data[:train_amount])
    test_data = jnp.array(data[train_amount:])
    return train_data, test_data


max_value = 97
train_data, test_data = create_dataset(0.99, max_value, lambda x, y: y)

seed = 123
key = jax.random.key(seed)

key, subkey = jax.random.split(key)

hidden_dims = [32, 32, 32, 32, max_value]

key, mlp = MLPModel.initialize(key, max_value, hidden_dims, jnp.tanh)

grad_fn = jax.value_and_grad(compute_loss, argnums=0)

lr = 0.1

loss_hist = deque(maxlen=10)

for epoch in range(10):
    key, subkey = jax.random.split(key)
    jax.random.permutation(subkey, train_data)

    progress_bar = tqdm.tqdm(
        batch_iterator(train_data, 32), total=len(train_data) // 32
    )

    for data, target in progress_bar:
        compute_loss(mlp, data, target)
        loss_value, grads = grad_fn(mlp, data, target)
        mlp = jax.tree_map(lambda p, g: p - lr * g, mlp, grads)

        loss_hist.append(loss_value)

        progress_bar.set_description(f"Loss: {np.mean(loss_hist):.2f}")

    correct = []
    for data, target in batch_iterator(test_data, 8):
        is_correct = compute_acc(mlp, data, target)
        correct.append(is_correct)

    correct = jnp.concatenate(correct)
    acc = 100.0 * correct.mean()
    print(f"Epoch {epoch + 1} Test Acc: {acc:.2f}%")
