from dataclasses import dataclass
import itertools
from itertools import product
import random
from typing import List, Any

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random


@register_pytree_node_class
@dataclass
class Layer:
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclass
class EmbeddingMatrix(Layer):
    M: jnp.array

    def __call__(self, indeces):
        return jnp.take(self.M, indeces, axis=0)

    @classmethod
    def initialize(cls, key, n, d):
        key, subkey = jax.random.split(key, 2)
        embedding_matrix = jax.random.normal(shape=(n, d), key=subkey)
        return key, cls(embedding_matrix)

    def tree_flatten(self):
        return ((self.M,), None)


@register_pytree_node_class
@dataclass
class Linear(Layer):
    w: jnp.array
    b: jnp.array
    act_fn: Any

    def __call__(self, x):
        h = jnp.matmul(x, self.w) + self.b
        return self.act_fn(h)

    @classmethod
    def initialize(cls, key, input_dim, output_dim, act):
        key, *subkey = jax.random.split(key, 3)
        weights = jax.random.normal(shape=(input_dim, output_dim), key=subkey[0])
        bias = jax.random.normal(shape=(output_dim,), key=subkey[1])
        return key, cls(weights, bias, act)

    def tree_flatten(self):
        return ((self.w, self.b), self.act_fn)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


@register_pytree_node_class
@dataclass
class Sequential(Layer):
    layers: List[Layer]

    def __call__(self, x):
        inp = x
        for layer in self.layers:
            inp = layer(inp)
        return inp

    def tree_flatten(self):
        return ((self.layers,), None)


def softmax(x):
    vals = jnp.exp(x)
    totals = jnp.sum(vals, axis=-1, keepdims=True)
    return vals / totals


def nll_loss(probs, target):
    log_probs = jnp.log(probs)
    ll = jnp.take_along_axis(log_probs, target.reshape(-1, 1), axis=1).mean()
    return -1.0 * ll


def forward(embed, mlp, data):
    left_embed = embed(data[:, 0])
    right_embed = embed(data[:, 1])
    target = data[:, 2]

    embeded = jnp.concatenate([left_embed, right_embed], axis=1)
    logits = mlp(embeded)
    probs = softmax(logits)
    loss_value = nll_loss(probs, target)
    return loss_value


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

embed_dim = 8
hidden_dim = 32


key, embedding_layer = EmbeddingMatrix.initialize(key, p, embed_dim)
key, layer_one = Linear.initialize(key, 2 * embed_dim, hidden_dim, jnp.tanh)
key, layer_two = Linear.initialize(key, hidden_dim, p, lambda x: x)
mlp = Sequential([layer_one, layer_two])


grad_fn = jax.value_and_grad(forward, argnums=(0, 1))

lr = 0.1
for epoch in range(10000):
    random.shuffle(train_data)
    for idx in range(0, len(train_data), 8):
        batch = train_data[idx : idx + 8]

        data = jnp.array(batch)

        loss_value, grads = grad_fn(embedding_layer, mlp, data)
        print(loss_value)

        embedding_layer = jax.tree_map(
            lambda p, g: p - lr * g, embedding_layer, grads[0]
        )
        mlp = jax.tree_map(lambda p, g: p - lr * g, mlp, grads[1])
