
import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp

def softmax(x: ArrayLike) -> Array:
    vals = jnp.exp(x)
    totals = jnp.sum(vals, axis=-1, keepdims=True)
    return vals / totals
