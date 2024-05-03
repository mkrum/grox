import jax
import jax.numpy as jnp
from grox.nn import softmax, affine, attention


def test_softmax():
    key = jax.random.key(123)
    input_vals = jax.random.normal(shape=(10, 2), key=key)
    output = softmax(input_vals)
    sums = jnp.sum(output, axis=-1)
    assert (sums - 1.0).mean() < 1e-6


def test_affine():
    key = jax.random.key(123)
    key, *subkeys = jax.random.split(key, 4)

    x = jax.random.normal(shape=(32, 8), key=subkeys[0])
    W = jax.random.normal(shape=(8, 16), key=subkeys[1])
    b = jax.random.normal(shape=(16,), key=subkeys[2])

    h = affine(x, W, b)
    assert h.shape == (32, 16)


def test_attention():
    key = jax.random.key(123)

    Q = jax.random.normal(shape=(32, 8), key=key)
    K = jax.random.normal(shape=(32, 8), key=key)
    V = jax.random.normal(shape=(32, 16), key=key)

    output = attention(Q, K, V)
    assert output.shape == (32, 16)
