
from dataclasses import dataclass
from typing import Any, List

import jax.random
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

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

@layer
class SimpleMLP(Layer):

    embedding: EmbeddingMatrix
    ffn_layers: Sequential

    @classmethod
    def initialize(cls, key, num_tokens, dim_list, act_fn):
        
        assert dim_list[0] % 2 == 0

        key, embed_layer = EmbeddingMatrix.initialize(key, num_tokens, dim_list[0] // 2)

        layers = []
        for idx in range(len(dim_list) - 1):

            if idx == len(dim_list) - 2:
                act_fn = lambda x: x

            key, layer = Linear.initialize(key, dim_list[idx], dim_list[idx + 1], act_fn)
            layers.append(layer)
        
        mlp = Sequential(layers)
        return key, cls(embed_layer, mlp)

    def __call__(self, data):
        left_embed = self.embedding(data[:, 0])
        right_embed = self.embedding(data[:, 1])
        embeded = jnp.concatenate([left_embed, right_embed], axis=1)
        return self.ffn_layers(embeded)

    def tree_flatten(self):
        return ((self.embedding, self.ffn_layers), None)
