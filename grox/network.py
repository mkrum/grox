from dataclasses import dataclass
from typing import Any, List
import numpy as np

import jax.random
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from grox.nn import attention


def xavier_init(key, shape):
    assert len(shape) == 2

    f_in = shape[0]
    f_out = shape[1]

    range_limit = 6 / np.sqrt(f_in + f_out)

    key, subkey = jax.random.split(key, 2)

    return jax.random.uniform(
        key=subkey, shape=shape, minval=-range_limit, maxval=range_limit
    )


def kaiming_init(key, shape):
    assert len(shape) == 2

    f_in = shape[0]
    f_out = shape[1]

    multiplier = np.sqrt(2) / f_in

    key, subkey = jax.random.split(key, 2)
    weights = jax.random.normal(key=subkey)

    final_weights = weights * multiplier

    return final_weights


def layer(x):
    return register_pytree_node_class(dataclass(x))


@layer
class Layer:
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def initialize(cls, key, init_fn, *args, **kwargs):
        ...


@layer
class EmbeddingMatrix(Layer):
    M: jnp.array

    @jax.jit
    def __call__(self, indeces):
        return jnp.take(self.M, indeces, axis=0)

    @classmethod
    def initialize(cls, key, n, d):
        key, subkey = jax.random.split(key, 2)
        embedding_matrix = jax.random.normal(subkey, (n, d))
        return key, cls(embedding_matrix)

    def tree_flatten(self):
        return ((self.M,), None)


@layer
class Affine(Layer):
    w: jnp.array
    b: jnp.array
    act_fn: Any

    @jax.jit
    def __call__(self, x):
        h = jnp.matmul(x, self.w.T) + self.b
        return self.act_fn(h)

    @classmethod
    def initialize(cls, key, input_dim, output_dim, act, init_type="normal"):
        init_fn = None

        if init_type == "normal":
            init_fn = jax.random.normal
        elif init_type == "xavier":
            init_fn = xavier_init
        elif init_type == "kaiming":
            init_fn = kaiming_init
        else:
            raise ValueError(
                f"Initialization {init_type} not one of: normal, xavier, kaiming"
            )

        key, subkey = jax.random.split(key, 2)
        weights = init_fn(subkey, shape=(output_dim, input_dim))

        # Initialize the bias at 0?
        bias = jnp.zeros(shape=(output_dim,))
        return key, cls(weights, bias, act)

    def tree_flatten(self):
        return ((self.w, self.b), self.act_fn)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


@layer
class Linear(Layer):
    w: jnp.array
    act_fn: Any

    @jax.jit
    def __call__(self, x):
        h = jnp.matmul(x, self.w)
        return self.act_fn(h)

    @classmethod
    def initialize(cls, key, input_dim, output_dim, act, init_type="normal"):
        init_fn = None

        if init_type == "normal":
            init_fn = jax.random.normal
        elif init_type == "xavier":
            init_fn = xavier_init
        elif init_type == "kaiming":
            init_fn = kaiming_init
        else:
            raise ValueError(
                f"Initialization {init_type} not one of: normal, xavier, kaiming"
            )

        key, subkey = jax.random.split(key, 2)
        weights = init_fn(subkey, shape=(input_dim, output_dim))

        return key, cls(weights, act)

    def tree_flatten(self):
        return ((self.w,), self.act_fn)

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
    ffn_layers: Sequential

    def __call__(self, x):
        return self.ffn_layers(x)

    @classmethod
    def initialize(cls, key, dim_list, act_fn=jnp.tanh, init_type="kaiming"):
        layers = []
        for idx in range(len(dim_list) - 1):
            if idx == len(dim_list) - 2:
                act_fn = lambda x: x

            key, layer = Linear.initialize(
                key, dim_list[idx], dim_list[idx + 1], act_fn, init_type=init_type
            )
            layers.append(layer)

        mlp = Sequential(layers)
        return key, cls(mlp)

    def tree_flatten(self):
        return ((self.ffn_layers), None)


@layer
class SimpleSelfAttentionBlock(Layer):
    W_proj: Any
    out_proj: Any

    def __call__(self, x):
        X = jnp.matmul(x, self.W_proj.T)
        Q, K, V = jnp.array_split(X, 3, axis=-1)
        output, weights = attention(Q, K, V)

        return jnp.matmul(output, self.out_proj.T)

    @classmethod
    def initialize(cls, key, input_dim):
        key, *subkeys = jax.random.split(key, 4)

        init_fn = kaiming_init

        W_proj = init_fn(subkeys[0], shape=(3 * input_dim, input_dim))

        return key, cls(W_proj)


@layer
class LayerNorm(Layer):
    bias: jnp.array
    gain: jnp.array

    def __call__(self, x):
        return layer_norm(x, gain, bias)

    @classmethod
    def initialize(cls, key, input_dim):
        bias = jnp.zeros(shape=(input_dim,))
        gain = jnp.zeroes(shape=(input_dim,))

        return key, cls(W_k, W_v, W_q)


@layer
class SimpleTransformerBlock(Layer):
    attn_block: SimpleSelfAttentionBlock
    mlp: SimpleMLP
    ln_1: LayerNorm
    ln_2: LayerNorm

    @classmethod
    def initialize(cls, key, input_dim, expanse_ratio):
        key, *subkeys = jax.random.split(key, 4)

        key, attn_block = SimpleSelfAttentionBlock.initialize(key, input_dim)
        key, mlp = SimpleMLP.initialize(
            key, [input_dim, int(expanse_ratio * input_dim), input_dim]
        )
        key, ln_1 = LayerNorm.initialize(key, input_dim)
        key, ln_2 = LayerNorm.initialize(key, input_dim)

        return key, cls(attn_block, mlp)

    def __call__(self, x):
        x = self.ln_1(x + self.attn_block(x))
        x = self.ln_2(x + self.mlp(x))
        return x
