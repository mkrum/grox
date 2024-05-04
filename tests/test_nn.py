import jax
import jax.numpy as jnp
import numpy as np
from grox.nn import softmax, affine, attention, layer_norm

import torch
import torch.nn.functional as F

ATOL=1e-6


def test_softmax():
    X = np.random.normal(size=(10, 2))

    X_torch = torch.tensor(X, dtype=torch.float32)
    output_torch = F.softmax(X_torch, dim=-1)

    X_jax = jnp.array(X, dtype='float32')
    output_grox = softmax(X_jax)

    assert np.allclose(output_grox, output_torch, atol=ATOL)


def test_affine():
    key = jax.random.key(123)
    key, *subkeys = jax.random.split(key, 4)

    x = jax.random.normal(shape=(32, 8), key=subkeys[0])
    W = jax.random.normal(shape=(8, 16), key=subkeys[1])
    b = jax.random.normal(shape=(16,), key=subkeys[2])

    h = affine(x, W, b)
    assert h.shape == (32, 16)


def test_attention():

    Q = np.random.normal(size=(32, 8))
    K = np.random.normal(size=(32, 8))
    V = np.random.normal(size=(32, 16))

    Q_torch = torch.tensor(Q, dtype=torch.float32)
    K_torch = torch.tensor(K, dtype=torch.float32)
    V_torch = torch.tensor(V, dtype=torch.float32)

    output_torch = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch, scale=1/np.sqrt(Q.shape[1]))

    Q_jax = jnp.array(Q, dtype='float32')
    K_jax = jnp.array(K, dtype='float32')
    V_jax = jnp.array(V, dtype='float32')

    output_grox = attention(Q_jax, K_jax, V_jax)

    assert np.allclose(output_grox, output_torch, atol=ATOL)

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

    assert np.allclose(out_grox, out_torch, atol=ATOL)
