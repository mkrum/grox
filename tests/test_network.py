import jax
import torch
import jax.nn
import torch.nn as nn
import numpy as np
import jax.numpy as jnp

from grox.network import (
    Affine,
    SimpleSelfAttentionBlock,
    SimpleTransformerBlock,
    LayerNorm,
    Sequential,
    Linear,
)

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


def test_attention_block():
    embed_dim = 8

    X = np.random.normal(size=(2, 2, embed_dim))
    W = np.random.normal(size=(3 * embed_dim, embed_dim))
    W2 = np.random.normal(size=(embed_dim, embed_dim))

    X_torch = torch.tensor(X, dtype=torch.float32)
    W_torch = torch.tensor(W, dtype=torch.float32)
    W2_torch = torch.tensor(W2, dtype=torch.float32)
    attn_torch = nn.MultiheadAttention(
        embed_dim, dropout=0.0, num_heads=1, bias=False, batch_first=True
    )

    attn_torch.in_proj_weight = nn.Parameter(W_torch)
    attn_torch.out_proj.weight = nn.Parameter(W2_torch)

    with torch.no_grad():
        out_torch, out_weights = attn_torch(X_torch, X_torch, X_torch)
        out_torch = np.array(out_torch)

    X_jax = jnp.array(X, dtype="float32")
    W_jax = jnp.array(W, dtype="float32")
    W2_jax = jnp.array(W2, dtype="float32")

    attn_grox = SimpleSelfAttentionBlock(W_jax, W2_jax)

    out_grox = np.array(attn_grox(X_jax))

    assert np.allclose(out_grox, out_torch, atol=ATOL)


def test_simple_transformer_block():
    embed_dim = 8
    X = np.random.normal(size=(2, 2, embed_dim))

    W = np.random.normal(size=(3 * embed_dim, embed_dim))
    W2 = np.random.normal(size=(embed_dim, embed_dim))

    lW1 = np.random.normal(size=(3 * embed_dim, embed_dim))
    lW2 = np.random.normal(size=(embed_dim, 3 * embed_dim))

    gain_1 = np.random.normal(size=(embed_dim,))
    gain_2 = np.random.normal(size=(embed_dim,))

    X_torch = torch.tensor(X, dtype=torch.float32)

    W_torch = torch.tensor(W, dtype=torch.float32)
    W2_torch = torch.tensor(W2, dtype=torch.float32)

    lW1_torch = torch.tensor(lW1, dtype=torch.float32)
    lW2_torch = torch.tensor(lW2, dtype=torch.float32)

    gain_1_torch = torch.tensor(gain_1, dtype=torch.float32)
    gain_2_torch = torch.tensor(gain_2, dtype=torch.float32)

    fn = nn.TransformerEncoderLayer(
        embed_dim,
        1,
        dim_feedforward=32,
        dropout=0.0,
        batch_first=True,
        bias=False,
        dtype=torch.float32,
        norm_first=False,
    )

    fn.self_attn.in_proj_weight = nn.Parameter(W_torch)
    fn.self_attn.out_proj.weight = nn.Parameter(W2_torch)

    fn.linear1.weight = nn.Parameter(lW1_torch)
    fn.linear2.weight = nn.Parameter(lW2_torch)

    fn.norm1.weight = nn.Parameter(gain_1_torch)
    fn.norm2.weight = nn.Parameter(gain_2_torch)

    with torch.no_grad():
        out_torch = fn(X_torch)
        out_torch = np.array(out_torch)

    X_jax = jnp.array(X, dtype="float32")

    W_jax = jnp.array(W, dtype="float32")
    W2_jax = jnp.array(W2, dtype="float32")

    lW1_jax = jnp.array(lW1, dtype="float32")
    lW2_jax = jnp.array(lW2, dtype="float32")

    gain_1_jax = jnp.array(gain_1, dtype="float32")
    gain_2_jax = jnp.array(gain_2, dtype="float32")

    attn_grox = SimpleSelfAttentionBlock(W_jax, W2_jax)

    norm1 = LayerNorm(gain_1_jax, None)
    norm2 = LayerNorm(gain_2_jax, None)

    mlp = Sequential([Linear(lW1_jax, jax.nn.relu), Linear(lW2_jax, lambda x: x)])

    block_grox = SimpleTransformerBlock(attn_grox, mlp, norm1, norm2)
    out_grox = block_grox(X_jax)

    assert np.allclose(out_grox, out_torch, atol=ATOL)
