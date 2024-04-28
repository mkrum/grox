from dataclasses import dataclass
import itertools
from itertools import product
import random

import jax.numpy as jnp
import jax.random
import jax.nn

train_per = 0.8
p = 7

modulo_addition = lambda x, y: x  # (x + y) % p

x = list(range(0, 7))
y = list(range(0, 7))
combos = list(itertools.product(x, y))

data = list(map(lambda x: (*x, modulo_addition(*x)), combos))

random.shuffle(data)

total = len(data)
train_amount = int(train_per * total)

train_data = data[:train_amount]
test_data = data[train_amount:]

seed = 123
key = jax.random.key(seed)

key, subkey = jax.random.split(key)

embed_dim = 8

embedding_matrix = jax.random.normal(shape=(p, embed_dim), key=key)


def embed(embedding_matrix, token_ids):
    return jnp.take(embedding_matrix, token_ids, axis=0)


key, subkey = jax.random.split(key)
l1_weights = jax.random.normal(shape=(2 * embed_dim, 32), key=key)
key, subkey = jax.random.split(key)
l1_bias = jax.random.normal(shape=(32,), key=key)

key, subkey = jax.random.split(key)
l2_weights = jax.random.normal(shape=(32, p), key=key)
l2_bias = jax.random.normal(shape=(p,), key=key)


def softmax(x):
    vals = jnp.exp(x)
    totals = jnp.sum(vals, axis=-1, keepdims=True)
    return vals / totals


def network(l1_weights, l1_bias, l2_weights, l2_bias, x):
    h1 = jnp.tanh(jnp.matmul(x, l1_weights) + l1_bias)

    out = jnp.matmul(h1, l2_weights) + l2_bias

    probs = softmax(out)
    return probs


def nll_loss(probs, target):
    log_probs = jnp.log(probs)
    ll = jnp.take_along_axis(log_probs, target.reshape(-1, 1), axis=1).mean()
    return -1.0 * ll


def forward(embedding_matrix, l1_weights, l1_bias, l2_weights, l2_bias, data):
    left_embed = embed(embedding_matrix, data[:, 0])
    right_embed = embed(embedding_matrix, data[:, 1])
    target = data[:, 2]

    embeded = jnp.concatenate([left_embed, right_embed], axis=1)
    probs = network(l1_weights, l1_bias, l2_weights, l2_bias, embeded)
    loss_value = nll_loss(probs, target)
    return loss_value


grad_fn = jax.value_and_grad(forward, argnums=(0, 1, 2, 3, 4))
params = (embedding_matrix, l1_weights, l1_bias, l2_weights, l2_bias)

lr = 0.1
for epoch in range(10000):
    random.shuffle(train_data)
    idx = 0
    for jdx in range(0, len(train_data), 8):
        batch = train_data[idx : idx + 8]

        data = jnp.array(batch)

        out = forward(*params, data)
        loss_value, grads = grad_fn(*params, data)
        print(loss_value)
        params = tuple(p - lr * g for p, g in zip(params, grads))
