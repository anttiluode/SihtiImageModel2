import jax
import jax.numpy as jnp
import numpy as np

from sihti2 import nets
from sihti2.layout import GaussLayout, Lift, chans, flip_tokens, layout, unchans, x16
from sihti2.patch import bands_of
from sihti2.sieve import sieve, unsieve


def test_sieve_lossless():
    x = np.random.default_rng(0).random((4, 3, 32, 32))
    assert np.abs(unsieve(sieve(x)) - x).max() < 1e-12


def test_chans_roundtrip():
    X = np.random.default_rng(1).random((5, 32, 32, 3)).astype(np.float32)
    assert np.array_equal(unchans(chans(X), 5), X)


def test_x16_is_core_plus_r3():
    X = np.random.default_rng(2).random((3, 32, 32, 3)).astype(np.float32)
    B = bands_of(chans(X))
    assert np.allclose(x16(X), unchans(B[3] + B[4], 3), atol=1e-6)
    assert np.allclose(unchans(sum(B), 3), X, atol=1e-5)


def test_layout_is_average_pool():
    X = np.random.default_rng(3).random((2, 32, 32, 3)).astype(np.float32)
    assert np.allclose(layout(X)[1, 2, 5], X[1, 8:12, 20:24].mean((0, 1)), atol=1e-6)


def test_flip_tokens_is_mirror():
    T = np.arange(64)[None]
    assert flip_tokens(T)[0, 0] == 7 and flip_tokens(flip_tokens(T)).tolist() == T.tolist()


def test_lift_fits_linear_target():
    rng = np.random.default_rng(4)
    L = rng.random((400, 8, 8, 3)).astype(np.float32)
    M = rng.standard_normal((192, 3072)).astype(np.float32) * 0.01
    Y = (L.reshape(400, -1) @ M).reshape(400, 32, 32, 3)
    assert np.abs(Lift().fit(L, Y, lam=1e-9)(L) - Y).max() < 1e-3


def test_gauss_layout_shapes_and_range():
    rng = np.random.default_rng(5)
    L = rng.random((200, 8, 8, 3)).astype(np.float32)
    g = GaussLayout().fit(L, np.repeat(np.arange(10), 20))
    S = g.sample(np.arange(10), rng)
    assert S.shape == (10, 8, 8, 3) and S.min() >= 0 and S.max() <= 1


def test_ar_is_causal():
    p = nets.ar_init(jax.random.PRNGKey(0), K=16, d=32, L=2)
    t = jnp.zeros((1, 64), jnp.int32)
    a = nets.ar_logits(p, jnp.array([3]), t)
    b = nets.ar_logits(p, jnp.array([3]), t.at[0, 40].set(5))
    # changing token 40 may only affect predictions at positions > 40
    assert np.allclose(a[0, :41], b[0, :41], atol=1e-5)
    assert not np.allclose(a[0, 41:], b[0, 41:])
