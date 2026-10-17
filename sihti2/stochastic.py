"""Arm R4 (PREREG_STOCHASTIC.md): the P3 mean plus a sample of each code's leftover residue.

residue patch = P3 mean(window) + scale(window) * L_k z,  z ~ N(0, I_16)
Overlaps: mean part averaged (as P3), sampled part summed / sqrt(count).
"""
import numpy as np

from .layout import chans, unchans
from .patch import G, NB, P, STRIDES, bands_of, nearest, normalise, overlap_add, patches, windows


def fit_residual_chol(pm, G_imgs, seed=1, per_image=10, batch=6000, shrink=1e-3):
    """G_imgs: (N,32,32) grey training channel images. Returns {band: (K,16,16) Cholesky factors}."""
    rng = np.random.default_rng(seed)
    out = {}
    for b in (2, 1, 0):
        a = pm.arms[b]
        K = len(a["C"])
        S = np.zeros((K, P * P, P * P))
        n = np.zeros(K)
        for s in range(0, len(G_imgs), batch):
            X = G_imgs[s:s + batch]
            Bs = bands_of(X)
            C = sum(Bs[j] for j in range(b + 1, NB))
            A, ii, jj = windows(C, STRIDES[b])
            V = patches(Bs[b], ii, jj)
            A = A.reshape(len(X), -1, G * G)
            V = V.reshape(len(X), -1, P * P)
            sel = np.stack([rng.choice(A.shape[1], per_image, replace=False) for _ in range(len(X))])
            A = np.take_along_axis(A, sel[..., None], 1).reshape(-1, G * G)
            V = np.take_along_axis(V, sel[..., None], 1).reshape(-1, P * P)
            An, sc = normalise(A)
            Vn = V / sc
            k = nearest(An, a["C"])
            x = np.concatenate([An, np.ones((len(An), 1), np.float32)], 1)
            R = Vn - np.einsum("nd,ndv->nv", x, a["W"][k])
            np.add.at(S, k, np.einsum("ni,nj->nij", R, R))
            np.add.at(n, k, 1)
        Lc = np.zeros((K, P * P, P * P), np.float32)
        for k in range(K):
            cov = S[k] / max(n[k], 1)
            cov += shrink * (np.trace(cov) / 16 + 1e-12) * np.eye(16)
            Lc[k] = np.linalg.cholesky(cov)
        out[b] = Lc
    return out


def predict_band_stochastic(pm, Lc, C, b, rng):
    N, H, W = C.shape
    A, ii, jj = windows(C, STRIDES[b])
    A = A.reshape(-1, G * G)
    a = pm.arms[b]
    An, sc = normalise(A)
    k = nearest(An, a["C"])
    x = np.concatenate([An, np.ones((len(An), 1), np.float32)], 1)
    mean = np.einsum("nd,ndv->nv", x, a["W"][k]) * sc
    z = rng.standard_normal((len(An), P * P)).astype(np.float32)
    noise = np.einsum("nij,nj->ni", Lc[b][k], z) * sc
    shape = (N, len(ii), len(jj), P * P)
    m = overlap_add(mean.reshape(shape), ii, jj, H, W)
    s = overlap_add(noise.reshape(shape), ii, jj, H, W)          # = sum / count
    counts = np.zeros((H, W), np.float32)
    for i in ii:
        for j in jj:
            counts[i:i + P, j:j + P] += 1
    return m + s * np.sqrt(counts)                                   # sum / sqrt(count)


def refine_stochastic(L, lift, pm, Lc, rng, return_stages=False):
    N = len(L)
    Cl = lift(L)
    cur = chans(Cl)
    stages, bands = [Cl], []
    for b in (2, 1, 0):
        r = predict_band_stochastic(pm, Lc, cur, b, rng)
        cur = cur + r
        bands.append(unchans(r, N))
        stages.append(unchans(cur, N))
    out = unchans(cur, N)
    return (out, stages, bands) if return_stages else out
