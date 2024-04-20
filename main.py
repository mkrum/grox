
from dataclasses import dataclass
import itertools
from itertools import product
import random

import jax.numpy as jnp
import jax.random

train_per = 0.8
p = 7

modulo_addition = lambda x, y: (x + y) % p

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

embed_dim = 2

embedding_matrix = jax.random.normal(shape=(p, embed_dim), key=key)

def embed(embedding_matrix, token_ids):
    return jnp.take(embedding_matrix, token_ids, axis=0)

batch = train_data[:8]

data = jnp.array(batch)
print(data[:, 0])
print(data)
print(embedding_matrix)

left_embed = embed(embedding_matrix, data[:, 0])
right_embed = embed(embedding_matrix, data[:, 1])
target = data[:, 2]

embeded = jnp.concatenate([left_embed, right_embed], axis=1)

key, subkey = jax.random.split(key)
l1_weights = jax.random.normal(shape=(2 * embed_dim, 8), key=key)
key, subkey = jax.random.split(key)
l1_bias = jax.random.normal(shape=(8, ), key=key)

key, subkey = jax.random.split(key)
l2_weights = jax.random.normal(shape=(8, p), key=key)
l2_bias = jax.random.normal(shape=(p, ),key=key)

def softmax(x):
    vals = jnp.exp(-x)
    return vals / jnp.sum(vals, axis=0)

def network(l1_weights, l1_bias, l2_weights, l2_bias, x):
    h1 = jnp.matmul(x, l1_weights) + l1_bias
    out = jnp.matmul(h1, l2_weights) + l2_bias
    probs = softmax(out)
    return probs

probs = network(l1_weights, l1_bias, l2_weights, l2_bias, embeded)
log_probs = jnp.log(probs)
nll = jnp.take(log_probs, target).mean()
