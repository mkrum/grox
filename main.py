from dataclasses import dataclass
import itertools
from itertools import product
import random
from typing import List, Any

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random

def layer(x):
    return register_pytree_node_class(dataclass(x))


@layer
class Layer:
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@layer
class EmbeddingMatrix(Layer):
    M: jnp.array
    
    @jax.jit
    def __call__(self, indeces):
        return jnp.take(self.M, indeces, axis=0)

    @classmethod
    def initialize(cls, key, n, d):
        key, subkey = jax.random.split(key, 2)
        embedding_matrix = jax.random.normal(shape=(n, d), key=subkey)
        return key, cls(embedding_matrix)

    def tree_flatten(self):
        return ((self.M,), None)


@layer
class Linear(Layer):
    w: jnp.array
    b: jnp.array
    act_fn: Any

    @jax.jit
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


@layer
class Sequential(Layer):
    layers: List[Layer]

    def __call__(self, x):
        inp = x
        for layer in self.layers:
            inp = layer(inp)
        return inp

    def tree_flatten(self):
        return ((self.layers,), None)


def batch_iterator(array, batchsize):
    for idx in range(0, len(array), batchsize):
        yield array[idx : idx + batchsize]


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

def compute_loss(embedding_layer, mlp, data):
    probs = forward(embedding_layer, mlp, data)
    target = data[:, 2]
    loss_value = nll_loss(probs, target)
    return loss_value

def compute_acc(embedding_layer, mlp, data):
    embeded = mlp_embed(embedding_layer, data)
    target = data[:, 2]
    logits = mlp(embeded)
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

modulo_addition = lambda x, y: x #(x + y) % p

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

embed_dim = 16
hidden_dim = 32


key, embedding_layer = EmbeddingMatrix.initialize(key, p, embed_dim)

hidden_dims = [2 * embed_dim, 32, 32, 32, p]

layers = []
for idx in range(len(hidden_dims) - 1):
    act_fn = jnp.tanh
    if idx == len(hidden_dims) - 2:
        act_fn = lambda x: x
    
    key, layer = Linear.initialize(key, hidden_dims[idx], hidden_dims[idx + 1], act_fn)

    layers.append(layer)

mlp = Sequential(layers)

grad_fn = jax.value_and_grad(compute_loss, argnums=(0, 1))

lr = 0.1
for epoch in range(10000):

    key, subkey = jax.random.split(key)
    jax.random.permutation(subkey, train_data)

    for batch in batch_iterator(train_data, 32):
        loss_value, grads = grad_fn(embedding_layer, mlp, batch)

        embedding_layer = jax.tree_map(
            lambda p, g: p - lr * g, embedding_layer, grads[0]
        )
        mlp = jax.tree_map(lambda p, g: p - lr * g, mlp, grads[1])

    print(loss_value)

    correct = []
    for batch in batch_iterator(test_data, 8):
        data = jnp.array(batch)
        is_correct = compute_acc(embedding_layer, mlp, data)
        correct.append(is_correct)

    correct = jnp.concatenate(correct)
    print("TEST ACC, ", correct.mean())

