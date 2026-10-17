"""Patch residue memory (copied unchanged from anttiluode/SihtiImageModel, sihti_model/patch.py).

Instead of addressing with the whole coarse image and storing whole-image residues, address with a
small window of the coarser reconstruction around one place and return the residue patch for that
place.  The same few hundred local rules serve every location and every image, so the memory has a
fixed size and does not care where in the frame a thing sits.

    band b at a place  <-  rule_k( 5x5 window of (core + ... + band b+1) around that place )

Arms (see PREREG_PATCH.md):
  P0  one linear map from the raw window          (a learned deconvolution filter)
  P1  k-NN over stored (window, patch) pairs     (example-based, grows with data; Freeman 2002 style)
  P2  K-means codebook, each code -> mean patch  (fixed size)
  P3  K-means codebook, each code -> local linear map   (fixed size; main arm)

Images here are 2D greyscale arrays; a batch is (N, H, W) of one shape.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .sieve import DEPTHS, sieve

G, P, STEP, EPS = 5, 4, 2, 0.02
STRIDES = {3: 4, 2: 2, 1: 1, 0: 1}        # window sampling stride per band (r3 coarse ... r0 fine)
NB = len(DEPTHS)                           # 4 residues + core


def bands_of(X):
    """X (N,H,W) -> list of 5 arrays (N,H,W), fine -> coarse."""
    return [b[:, 0] for b in sieve(X[:, None], DEPTHS, np.float32)]


def _grid(L):
    g = list(range(0, L - P + 1, STEP))
    if g[-1] != L - P:
        g.append(L - P)
    return np.array(g)


def windows(C, s):
    """C (N,H,W) coarser reconstruction -> raw windows (N, ni, nj, 25)."""
    N, H, W = C.shape
    m = (G // 2) * s + P
    Cp = np.pad(C, ((0, 0), (m, m), (m, m)), mode="edge")
    ii, jj = _grid(H), _grid(W)
    off = (np.arange(G) - G // 2) * s
    r = ii[:, None] + 2 + m + off[None]                   # (ni, G)
    c = jj[:, None] + 2 + m + off[None]                   # (nj, G)
    w = Cp[:, r[:, None, :, None], c[None, :, None, :]]   # (N, ni, nj, G, G)
    return w.reshape(N, len(ii), len(jj), G * G), ii, jj


def patches(B, ii, jj):
    """B (N,H,W) band -> (N, ni, nj, 16)."""
    v = sliding_window_view(B, (P, P), axis=(1, 2))[:, ii][:, :, jj]
    return v.reshape(B.shape[0], len(ii), len(jj), P * P)


def overlap_add(V, ii, jj, H, W):
    N = V.shape[0]
    out = np.zeros((N, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    V = V.reshape(N, len(ii), len(jj), P, P)
    for a, i in enumerate(ii):
        for b, j in enumerate(jj):
            out[:, i:i + P, j:j + P] += V[:, a, b]
            cnt[i:i + P, j:j + P] += 1
    return out / cnt


def normalise(A):
    mu = A.mean(-1, keepdims=True)
    sc = A.std(-1, keepdims=True) + EPS
    return (A - mu) / sc, sc


def _kmeans(X, K, rng, iters=25, batch=60000):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, max_iter=iters,
                         random_state=int(rng.integers(1 << 30)))
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


def nearest(X, Cn, chunk=50000):
    c2 = (Cn ** 2).sum(1)
    out = np.empty(len(X), np.int64)
    for s in range(0, len(X), chunk):
        q = X[s:s + chunk]
        out[s:s + chunk] = np.argmin(c2[None] - 2 * q @ Cn.T, 1)
    return out


class PatchMemory:
    def __init__(self, K=512, bank=50000, per_image=300, seed=0):
        self.K, self.bank_n, self.per_image = K, bank, per_image
        self.rng = np.random.default_rng(seed)
        self.arms = {}

    # ---- fitting -----------------------------------------------------------------
    def _samples(self, batches):
        """batches: iterable of (N,H,W) arrays.  Returns {band: (raw windows A, patches V)}."""
        acc = {b: ([], []) for b in (3, 2, 1, 0)}
        for X in batches:
            Bs = bands_of(X)
            for b in (3, 2, 1, 0):
                C = sum(Bs[j] for j in range(b + 1, NB))
                A, ii, jj = windows(C, STRIDES[b])
                V = patches(Bs[b], ii, jj)
                A = A.reshape(len(X), -1, G * G)
                V = V.reshape(len(X), -1, P * P)
                n = A.shape[1]
                k = min(self.per_image, n)
                sel = np.stack([self.rng.choice(n, k, replace=False) for _ in range(len(X))])
                acc[b][0].append(np.take_along_axis(A, sel[..., None], 1).reshape(-1, G * G))
                acc[b][1].append(np.take_along_axis(V, sel[..., None], 1).reshape(-1, P * P))
        return {b: (np.concatenate(a), np.concatenate(v)) for b, (a, v) in acc.items()}

    def fit(self, batches_fn, verbose=True):
        """batches_fn(): returns an iterable of (N,H,W) training batches."""
        S = self._samples(batches_fn())
        for b in (3, 2, 1, 0):
            A, V = S[b]
            arm = {}
            # P0: plain linear map on the raw window
            A1 = np.concatenate([A, np.ones((len(A), 1), np.float32)], 1).astype(np.float64)
            Gm = A1.T @ A1
            Gm[np.diag_indices(len(Gm))] += 1e-4 * np.trace(Gm) / len(Gm)
            arm["W0"] = np.linalg.solve(Gm, A1.T @ V).astype(np.float32)
            # normalised
            An, sc = normalise(A)
            Vn = V / sc
            Cn = _kmeans(An, self.K, self.rng)
            lab = nearest(An, Cn)
            means = np.zeros((self.K, P * P), np.float32)
            Wk = np.zeros((self.K, G * G + 1, P * P), np.float32)
            An1 = np.concatenate([An, np.ones((len(An), 1), np.float32)], 1).astype(np.float64)
            for k in range(self.K):
                m = lab == k
                if m.sum() == 0:
                    continue
                means[k] = Vn[m].mean(0)
                if m.sum() >= 40:
                    X = An1[m]
                    Gk = X.T @ X
                    Gk[np.diag_indices(len(Gk))] += 1e-2 * np.trace(Gk) / len(Gk)
                    Wk[k] = np.linalg.solve(Gk, X.T @ Vn[m])
                else:
                    Wk[k, -1] = means[k]
            arm["C"], arm["mean"], arm["W"] = Cn, means, Wk
            sel = self.rng.choice(len(An), min(self.bank_n, len(An)), replace=False)
            arm["bankA"], arm["bankV"] = An[sel].astype(np.float32), Vn[sel].astype(np.float32)
            self.arms[b] = arm
            if verbose:
                print(f"  patch band r{b}: {len(A)} samples, codebook {self.K}")
        return self

    def size(self):
        """Stored numbers for the fixed-size arms (P2, P3) and for P1's bank."""
        p2 = sum(a["C"].size + a["mean"].size for a in self.arms.values())
        p3 = sum(a["C"].size + a["W"].size for a in self.arms.values())
        p1 = sum(a["bankA"].size + a["bankV"].size for a in self.arms.values())
        p0 = sum(a["W0"].size for a in self.arms.values())
        return {"P0": p0, "P1": p1, "P2": p2, "P3": p3}

    # ---- prediction ----------------------------------------------------------------
    def predict_band(self, C, b, arm="P3", topk=8, chunk=40000):
        N, H, W = C.shape
        A, ii, jj = windows(C, STRIDES[b])
        A = A.reshape(-1, G * G)
        a = self.arms[b]
        if arm == "P0":
            V = np.concatenate([A, np.ones((len(A), 1), np.float32)], 1) @ a["W0"]
        else:
            An, sc = normalise(A)
            if arm == "P1":
                V = np.empty((len(A), P * P), np.float32)
                B2 = (a["bankA"] ** 2).sum(1)
                for s in range(0, len(A), 4000):
                    q = An[s:s + 4000]
                    d2 = (q ** 2).sum(1)[:, None] + B2[None] - 2 * q @ a["bankA"].T
                    nn = np.argpartition(d2, topk - 1, 1)[:, :topk]
                    dn = np.take_along_axis(d2, nn, 1)
                    w = np.exp(-(dn - dn.min(1, keepdims=True)) / (dn.mean(1, keepdims=True) + 1e-9))
                    w /= w.sum(1, keepdims=True)
                    V[s:s + 4000] = np.einsum("nk,nkd->nd", w, a["bankV"][nn])
            else:
                k = nearest(An, a["C"])
                if arm == "P2":
                    V = a["mean"][k]
                else:
                    V = np.empty((len(A), P * P), np.float32)
                    for s in range(0, len(A), chunk):
                        x = np.concatenate([An[s:s + chunk], np.ones((len(An[s:s + chunk]), 1), np.float32)], 1)
                        V[s:s + chunk] = np.einsum("nd,ndv->nv", x, a["W"][k[s:s + chunk]])
            V = V * sc
        return overlap_add(V.reshape(N, len(ii), len(jj), P * P), ii, jj, H, W)

    def complete(self, Bs, keep=2, arm="P3"):
        """Bs: true bands (fine->coarse). Keeps the coarsest `keep`, predicts the rest coarse->fine.
        Returns (image, predicted bands list)."""
        out = [None] * NB
        for j in range(NB - keep, NB):
            out[j] = Bs[j]
        for b in range(NB - keep - 1, -1, -1):
            C = sum(out[j] for j in range(b + 1, NB))
            out[b] = self.predict_band(C, b, arm)
        return sum(out), out
