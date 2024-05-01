
import jax
import jax.numpy as jnp
from grox.nn import softmax


def test_softmax():
    key = jax.random.key(123)
    input_vals = jax.random.normal(shape=(10, 2), key=key)
    output = softmax(input_vals)
    sums = jnp.sum(output, axis=-1)
    assert (sums - 1.0).mean() < 1e-6
    
