"""Layout, palette, Gaussian layout arm (LG), lift, and the full refine pipeline.

Images here are float32 (N,32,32,3) in [0,1]. A layout is (N,8,8,3): a 4x4 average pool.
"""
import numpy as np
from PIL import Image

from .patch import NB, bands_of

S = 8                      # layout side
F = 32 // S                # pool factor


def to_float(X):
    return X.astype(np.float32) / 255.0


def layout(X):
    N = len(X)
    return X.reshape(N, S, F, S, F, 3).mean((2, 4))


def chans(X):
    """(N,32,32,3) -> (3N,32,32) grey stack, channel-major per image."""
    return X.transpose(0, 3, 1, 2).reshape(-1, X.shape[1], X.shape[2])


def unchans(Y, N):
    return Y.reshape(N, 3, Y.shape[1], Y.shape[2]).transpose(0, 2, 3, 1)


def x16(X):
    """Sihti coarse state core + r3 (diffusion depth 16), per channel. (N,32,32,3)."""
    B = bands_of(chans(X))
    return unchans(B[3] + B[4], len(X))


# ------------------------------------------------------------------ palette (for LA)
class Palette:
    def __init__(self, K=512, seed=0):
        self.K, self.seed = K, seed

    def fit(self, L):
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=self.K, batch_size=16384, n_init=3, max_iter=60,
                             random_state=self.seed).fit(L.reshape(-1, 3))
        self.C = km.cluster_centers_.astype(np.float32)
        return self

    def encode(self, L):
        x = L.reshape(-1, 3)
        c2 = (self.C ** 2).sum(1)
        out = np.empty(len(x), np.int32)
        for s in range(0, len(x), 200000):
            q = x[s:s + 200000]
            out[s:s + 200000] = np.argmin(c2[None] - 2 * q @ self.C.T, 1)
        return out.reshape(len(L), S * S)

    def decode(self, T):
        return self.C[T].reshape(len(T), S, S, 3)


def flip_tokens(T):
    return T.reshape(-1, S, S)[:, :, ::-1].reshape(len(T), S * S)


# ------------------------------------------------------------------ LG: Gaussian layout arm
class GaussLayout:
    def fit(self, L, y, shrink=1e-3):
        self.mu, self.ch = [], []
        for c in range(10):
            Z = L[y == c].reshape(-1, S * S * 3).astype(np.float64)
            mu = Z.mean(0)
            Cv = np.cov(Z, rowvar=False)
            Cv += shrink * np.trace(Cv) / len(Cv) * np.eye(len(Cv))
            self.mu.append(mu)
            self.ch.append(np.linalg.cholesky(Cv))
        return self

    def sample(self, cls, rng):
        out = np.empty((len(cls), S * S * 3))
        for i, c in enumerate(cls):
            out[i] = self.mu[c] + self.ch[c] @ rng.standard_normal(S * S * 3)
        return np.clip(out, 0, 1).reshape(len(cls), S, S, 3).astype(np.float32)


# ------------------------------------------------------------------ lift L -> x16
class Lift:
    """One ridge map from the 192 layout numbers (+1) to the 3072 numbers of x16."""

    def fit(self, L, X16, lam=1e-3):
        A = np.concatenate([L.reshape(len(L), -1), np.ones((len(L), 1), np.float32)], 1).astype(np.float64)
        Gm = A.T @ A
        Gm[np.diag_indices(len(Gm))] += lam * np.trace(Gm) / len(Gm)
        self.W = np.linalg.solve(Gm, A.T @ X16.reshape(len(X16), -1)).astype(np.float32)
        return self

    def __call__(self, L):
        A = np.concatenate([L.reshape(len(L), -1), np.ones((len(L), 1), np.float32)], 1)
        return (A @ self.W).reshape(len(L), 32, 32, 3)


def bicubic(L):
    out = np.empty((len(L), 32, 32, 3), np.float32)
    for i, l in enumerate(L):
        im = Image.fromarray(np.clip(l * 255 + 0.5, 0, 255).astype(np.uint8))
        out[i] = np.asarray(im.resize((32, 32), Image.BICUBIC), np.float32) / 255
    return out


def refine(L, lift, pm, arm, return_stages=False):
    """Layout -> image. arm in R0 (lift), R1 (bicubic), R2 (lift+P0), R3 (lift+P3)."""
    if arm == "R1":
        return bicubic(L)
    C = lift(L)
    if arm == "R0":
        return (C, [C]) if return_stages else C
    a = {"R2": "P0", "R3": "P3"}[arm]
    N = len(L)
    cur = chans(C)
    stages, bands = [C], []
    for b in (2, 1, 0):
        r = pm.predict_band(cur, b, a)
        cur = cur + r
        bands.append(unchans(r, N))
        stages.append(unchans(cur, N))
    out = unchans(cur, N)
    return (out, stages, bands) if return_stages else out
