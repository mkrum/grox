import tqdm
import random
import itertools
from dataclasses import dataclass
from itertools import product
from typing import List, Any

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random

from grox.network import SimpleMLP


def batch_iterator(array, batchsize):
    for idx in range(0, len(array), batchsize):
        data = array[idx : idx + batchsize]
        inputs = data[:, :2]
        targets = data[:, 3]
        yield (inputs, targets)


def softmax(x):
    vals = jnp.exp(x)
    totals = jnp.sum(vals, axis=-1, keepdims=True)
    return vals / totals


def nll_loss(probs, target):
    log_probs = jnp.log(probs)
    ll = jnp.take_along_axis(log_probs, target.reshape(-1, 1), axis=1).mean()
    return -1.0 * ll


def forward(embedding_layer, mlp, data):
    embeded = mlp_embed(embedding_layer, data)
    logits = mlp(embeded)
    probs = softmax(logits)
    return probs


def compute_loss(mlp, data, target):
    probs = softmax(mlp(data))
    loss_value = nll_loss(probs, target)
    return loss_value


def compute_acc(mlp, data, target):
    logits = mlp(data)
    probs = softmax(logits)
    preds = jnp.argmax(probs, axis=1)
    return preds == target


def mlp_embed(embedding_layer, data):
    left_embed = embedding_layer(data[:, 0])
    right_embed = embedding_layer(data[:, 1])
    embeded = jnp.concatenate([left_embed, right_embed], axis=1)
    return embeded


train_per = 0.99
p = 100

# Dummy Target for testing
modulo_addition = lambda x, y: x 

x = list(range(0, p))
y = list(range(0, p))
combos = list(itertools.product(x, y))

data = list(map(lambda x: (*x, modulo_addition(*x)), combos))

random.shuffle(data)

total = len(data)
train_amount = int(train_per * total)

train_data = jnp.array(data[:train_amount])
test_data = jnp.array(data[train_amount:])

seed = 123
key = jax.random.key(seed)

key, subkey = jax.random.split(key)

hidden_dims = [32, 32, 32, 32, p]

key, mlp = SimpleMLP.initialize(key, p, hidden_dims, jnp.tanh)

grad_fn = jax.value_and_grad(compute_loss, argnums=0)

lr = 0.1
for epoch in range(10):
    key, subkey = jax.random.split(key)
    jax.random.permutation(subkey, train_data)

    for data, target in tqdm.tqdm(batch_iterator(train_data, 32), total=len(train_data) // 32):
        compute_loss(mlp, data, target)
        loss_value, grads = grad_fn(mlp, data, target)
        mlp = jax.tree_map(lambda p, g: p - lr * g, mlp, grads)

    print(loss_value)

    correct = []
    for data, target in batch_iterator(test_data, 8):
        is_correct = compute_acc(mlp, data, target)
        correct.append(is_correct)

    correct = jnp.concatenate(correct)
    print("TEST ACC ", correct.mean())
