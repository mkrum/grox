import jax
import torch
import torch.nn as nn
import numpy as np
import jax.numpy as jnp

from grox.network import Affine

ATOL = 1e-6


def test_affine():
    X = np.random.normal(size=(32, 16))
    w = np.random.normal(size=(8, 16))
    b = np.random.normal(size=(8,))

    key = jax.random.key(123)

    torch_layer = nn.Linear(16, 8)

    X_torch = torch.tensor(X, dtype=torch.float32)

    torch_layer.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))
    torch_layer.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    with torch.no_grad():
        out_torch = np.array(torch_layer(X_torch))

    X_jax = jnp.array(X, dtype="float32")
    w_jax = jnp.array(w, dtype="float32")
    b_jax = jnp.array(b, dtype="float32")

    grox_layer = Affine(w_jax, b_jax, lambda x: x)

    out_grox = np.array(grox_layer(X_jax))

    assert np.allclose(out_grox, out_torch, atol=ATOL)
