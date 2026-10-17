"""The only backprop in this repo: the layout model (LA) and the judge.

Pure JAX, no framework, so the same arithmetic can be re-implemented in the browser page
(web/model.js) and checked against this file.

Layout model: a class-conditional autoregressive transformer over 64 palette tokens
(an 8x8 layout in raster order). Pre-LayerNorm blocks, tanh-GELU, learned positions.
    position 0 input  = class embedding + pos[0]          -> predicts token 0
    position i input  = token embedding(t[i-1]) + pos[i]  -> predicts token i

Judge: a small CNN, 32-32-pool-64-64-pool-128-global-mean-10.
"""
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------ shared optimiser
def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": 0}


def adam_step(params, grads, st, lr, b1=0.9, b2=0.99, eps=1e-8, wd=0.0):
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, st["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, st["v"], grads)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * ((m / c1) / (jnp.sqrt(v / c2) + eps) + wd * p * (p.ndim > 1)),
        params, m, v)
    return params, {"m": m, "v": v, "t": t}


def save(obj, path):
    pickle.dump(jax.tree_util.tree_map(np.asarray, obj), open(path, "wb"))


def load(path):
    return jax.tree_util.tree_map(jnp.asarray, pickle.load(open(path, "rb")))


# ------------------------------------------------------------------ layout transformer
SEQ, NCLS = 64, 10


def ar_init(key, K=512, d=128, L=4, H=4):
    ks = iter(jax.random.split(key, 8 + 6 * L))
    n = lambda shape, s: s * jax.random.normal(next(ks), shape)
    p = {"tok": n((K, d), 0.02), "cls": n((NCLS, d), 0.02), "pos": n((SEQ, d), 0.02),
         "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d), "head": n((d, K), 0.02), "blocks": []}
    for _ in range(L):
        p["blocks"].append({
            "ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d),
            "qkv": n((d, 3 * d), 0.02), "qkv_b": jnp.zeros(3 * d),
            "o": n((d, d), 0.02 / math.sqrt(2 * L)), "o_b": jnp.zeros(d),
            "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
            "w1": n((d, 4 * d), 0.02), "b1": jnp.zeros(4 * d),
            "w2": n((4 * d, d), 0.02 / math.sqrt(2 * L)), "b2": jnp.zeros(d)})
    return p


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b


def _gelu(x):
    return 0.5 * x * (1 + jnp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def ar_logits(p, cls, toks, H=4):
    """cls (B,), toks (B,64) -> logits (B,64,K) where position i predicts toks[:, i]."""
    B = toks.shape[0]
    x = jnp.concatenate([p["cls"][cls][:, None], p["tok"][toks[:, :-1]]], 1) + p["pos"][None]
    d = x.shape[-1]
    hd = d // H
    mask = jnp.tril(jnp.ones((SEQ, SEQ), bool))
    for blk in p["blocks"]:
        h = _ln(x, blk["ln1_g"], blk["ln1_b"])
        qkv = h @ blk["qkv"] + blk["qkv_b"]
        q, k, v = jnp.split(qkv, 3, -1)
        q = q.reshape(B, SEQ, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, SEQ, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, SEQ, H, hd).transpose(0, 2, 1, 3)
        a = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
        a = jnp.where(mask, a, -1e9)
        a = jax.nn.softmax(a, -1)
        y = (a @ v).transpose(0, 2, 1, 3).reshape(B, SEQ, d)
        x = x + y @ blk["o"] + blk["o_b"]
        h = _ln(x, blk["ln2_g"], blk["ln2_b"])
        x = x + _gelu(h @ blk["w1"] + blk["b1"]) @ blk["w2"] + blk["b2"]
    x = _ln(x, p["lnf_g"], p["lnf_b"])
    return x @ p["head"]


def ar_loss(p, cls, toks):
    lg = ar_logits(p, cls, toks)
    lp = jax.nn.log_softmax(lg, -1)
    return -jnp.take_along_axis(lp, toks[..., None], -1).mean()


def ar_sample(p, cls, key, temperature=1.0):
    """cls (B,) -> toks (B,64). Plain ancestral sampling from the full distribution."""
    B = cls.shape[0]
    toks = jnp.zeros((B, SEQ), jnp.int32)
    f = jax.jit(ar_logits)
    for i in range(SEQ):
        key, sk = jax.random.split(key)
        lg = f(p, cls, toks)[:, i] / temperature
        toks = toks.at[:, i].set(jax.random.categorical(sk, lg, -1))
    return np.asarray(toks)


# ------------------------------------------------------------------ judge CNN
def cnn_init(key):
    ks = iter(jax.random.split(key, 8))
    he = lambda shape: jax.random.normal(next(ks), shape) * math.sqrt(2.0 / np.prod(shape[:-1]))
    chans = [(3, 32), (32, 32), (32, 64), (64, 64), (64, 128)]
    return {"convs": [{"w": he((3, 3, a, b)), "b": jnp.zeros(b)} for a, b in chans],
            "fc": he((128, 10)), "fc_b": jnp.zeros(10)}


def _conv(x, w, b):
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return jax.nn.relu(y + b)


def _pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")


MEAN = jnp.array([0.49, 0.48, 0.45])
STD = jnp.array([0.25, 0.24, 0.26])


def cnn_features(p, x):
    """x (B,32,32,3) float in [0,1] -> (B,128) pooled features."""
    x = (x - MEAN) / STD
    c = p["convs"]
    x = _conv(x, c[0]["w"], c[0]["b"])
    x = _pool(_conv(x, c[1]["w"], c[1]["b"]))
    x = _conv(x, c[2]["w"], c[2]["b"])
    x = _pool(_conv(x, c[3]["w"], c[3]["b"]))
    x = _conv(x, c[4]["w"], c[4]["b"])
    return x.mean((1, 2))


def cnn_logits(p, x):
    return cnn_features(p, x) @ p["fc"] + p["fc_b"]


def cnn_loss(p, x, y):
    lp = jax.nn.log_softmax(cnn_logits(p, x), -1)
    return -jnp.take_along_axis(lp, y[:, None], -1).mean()
