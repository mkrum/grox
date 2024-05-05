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
def attention(Q: ArrayLike, K: ArrayLike, V: ArrayLike) -> Array:
    # Handle multi-headed
    K_T = jnp.moveaxis(K, -2, -1)
    A = jnp.matmul(Q, K_T) / np.sqrt(Q.shape[-1])
    weights = softmax(A)
    return jnp.matmul(weights, V), weights


# @jax.jit
def layer_norm(
    X: ArrayLike, gain: ArrayLike, bias: ArrayLike, eps: float = 1e-5
) -> Array:
    X_mean = jnp.mean(X, axis=-1, keepdims=True)
    X_var = jnp.var(X, axis=-1, keepdims=True)

    X_norm = (X - X_mean) / jnp.sqrt(X_var + eps)

    if not (gain is None):
        X_norm = gain * X_norm

    if not (bias is None):
        X_norm = X_norm + bias

    return X_norm
