import jax
from jax import Array
import numpy as np
from jax.typing import ArrayLike
import jax.numpy as jnp


@jax.jit
def softmax(x: ArrayLike) -> Array:
    vals = jnp.exp(x)
    totals = jnp.sum(vals, axis=-1, keepdims=True)
    return vals / totals


@jax.jit
def affine(x: ArrayLike, W: ArrayLike, b: ArrayLike) -> Array:
    h = jnp.matmul(x, W) + b
    return h


@jax.jit
def attention(Q: ArrayLike, K: ArrayLike, V: ArrayLike) -> Array:
    A = jnp.matmul(Q, K.T) / np.sqrt(Q.shape[1])
    weights = softmax(A)
    return jnp.matmul(weights, V)
