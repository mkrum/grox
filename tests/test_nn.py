import jax
import jax.numpy as jnp
import numpy as np
from grox.nn import softmax, affine, attention, layer_norm

import torch
import torch.nn.functional as F


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
    
    # multi-headed
    Q = jax.random.normal(shape=(32, 8, 4), key=key)
    K = jax.random.normal(shape=(32, 8, 4), key=key)
    V = jax.random.normal(shape=(32, 8, 2), key=key)

    output = attention(Q, K, V)
    assert output.shape == (32, 8, 2)

def test_layer_norm():

    X = np.random.normal(size=(32, 16, 8))
    gain = np.random.normal(size=(8, ))
    bias = np.random.normal(size=(8, ))

    X_torch = torch.tensor(X,dtype=torch.float32)
    gain_torch = torch.tensor(gain, dtype=torch.float32)
    bias_torch = torch.tensor(bias, dtype=torch.float32)

    out_torch = F.layer_norm(X_torch, (8, ), weight=gain_torch, bias=bias_torch)

    X_jax = jnp.array(X, dtype='float32')
    gain_jax = jnp.array(gain, dtype='float32')
    bias_jax = jnp.array(bias, dtype='float32')
    out_grox = layer_norm(X_jax, gain_jax, bias_jax)
    
    out_grox = np.array(out_grox)
    out_torch = np.array(out_torch)

    assert np.allclose(out_grox, out_torch, atol=1e-6)
