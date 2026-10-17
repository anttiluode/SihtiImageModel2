"""Fit everything (no gates here). Writes results/cache/*.pkl and results/fit_log.txt.

    python fit_models.py            # palette, lift, Gaussian layouts, patch memory, judge, layout transformer
    python fit_models.py --only ar  # just one part: palette|closed|judge|ar

The 5 000 validation images (last 500 per class of train) are used only to pick the layout
transformer's checkpoint. The test set is not touched here.
"""
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from sihti2 import nets
from sihti2.data import load
from sihti2.layout import GaussLayout, Lift, Palette, chans, flip_tokens, layout, to_float, x16
from sihti2.patch import PatchMemory

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "results", "cache")
os.makedirs(CACHE, exist_ok=True)
LOG = open(os.path.join(HERE, "results", "fit_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def split():
    X, y, src = load("train")
    val = np.zeros(len(y), bool)
    for c in range(10):
        val[np.where(y == c)[0][-500:]] = True
    return to_float(X[~val]), y[~val], to_float(X[val]), y[val], src


def cache(name):
    return os.path.join(CACHE, name)


def fit_palette(Xtr, Xva):
    t = time.time()
    pal = Palette(512).fit(layout(Xtr))
    err = np.sqrt(((pal.decode(pal.encode(layout(Xva))) - layout(Xva)) ** 2).mean())
    log(f"palette 512: fit {time.time() - t:.0f}s, val layout RMS quantisation error {err:.4f}")
    pickle.dump(pal, open(cache("palette.pkl"), "wb"))


def fit_closed(Xtr, ytr, Xva):
    t = time.time()
    Ltr = layout(Xtr)
    lift = Lift().fit(Ltr, x16(Xtr))
    Xv16 = x16(Xva)
    e = lift(layout(Xva)) - Xv16
    log(f"lift: val PSNR of lifted x16 vs true x16 {10 * np.log10(1 / (e ** 2).mean()):.2f} dB ({time.time() - t:.0f}s)")
    pickle.dump(lift, open(cache("lift.pkl"), "wb"))
    pickle.dump(GaussLayout().fit(Ltr, ytr), open(cache("gauss.pkl"), "wb"))
    t = time.time()
    G = chans(Xtr)
    pm = PatchMemory(K=512, per_image=10, seed=0).fit(lambda: (G[i:i + 6000] for i in range(0, len(G), 6000)))
    log(f"patch memory: fit {time.time() - t:.0f}s on {len(G)} grey channel images, sizes {pm.size()}")
    pickle.dump(pm, open(cache("patch.pkl"), "wb"))


def fit_judge(Xtr, ytr, Xva, yva, epochs=10, minutes=30, bs=128):
    key = jax.random.PRNGKey(1)
    p = nets.cnn_init(key)
    st = nets.adam_init(p)
    steps = epochs * (len(Xtr) // bs)

    @jax.jit
    def step(p, st, x, y, lr):
        l, g = jax.value_and_grad(nets.cnn_loss)(p, x, y)
        p, st = nets.adam_step(p, g, st, lr, wd=5e-4)
        return p, st, l

    acc = jax.jit(lambda p, x: jnp.argmax(nets.cnn_logits(p, x), -1))
    rng = np.random.default_rng(1)
    t0, k = time.time(), 0
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        ls = []
        for s in range(0, len(perm) - bs + 1, bs):
            i = perm[s:s + bs]
            x = Xtr[i].copy()
            fl = rng.random(bs) < 0.5
            x[fl] = x[fl, :, ::-1]
            lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * k / steps))
            p, st, l = step(p, st, jnp.asarray(x), jnp.asarray(ytr[i]), lr)
            ls.append(float(l))
            k += 1
        va = np.mean(np.concatenate([np.asarray(acc(p, jnp.asarray(Xva[s:s + 1000]))) for s in range(0, len(Xva), 1000)]) == yva)
        log(f"judge epoch {ep + 1}: loss {np.mean(ls):.3f} val acc {va:.4f} ({time.time() - t0:.0f}s)")
        if time.time() - t0 > minutes * 60:
            log("judge: time budget reached")
            break
    nets.save(p, cache("judge.pkl"))


def fit_ar(Xtr, ytr, Xva, yva, epochs=15, minutes=75, bs=128):
    pal = pickle.load(open(cache("palette.pkl"), "rb"))
    Ttr, Tva = pal.encode(layout(Xtr)), pal.encode(layout(Xva))
    key = jax.random.PRNGKey(2)
    p = nets.ar_init(key, K=pal.K)
    st = nets.adam_init(p)
    nper = len(Ttr) // bs
    steps = epochs * nper

    @jax.jit
    def step(p, st, c, t, lr):
        l, g = jax.value_and_grad(nets.ar_loss)(p, c, t)
        p, st = nets.adam_step(p, g, st, lr, wd=0.01)
        return p, st, l

    vloss = jax.jit(nets.ar_loss)
    t0, k, best, start, spent = time.time(), 0, 1e9, 0, 0.0
    resume = cache("ar_resume.pkl")
    if os.path.exists(resume):          # survive a machine restart: continue from the last finished epoch
        r = pickle.load(open(resume, "rb"))
        p = jax.tree_util.tree_map(jnp.asarray, r["p"])
        best, start, spent = float(r["best"]), int(r["epoch"]), float(r["spent"])
        st = {"m": jax.tree_util.tree_map(jnp.asarray, r["st"]["m"]),
              "v": jax.tree_util.tree_map(jnp.asarray, r["st"]["v"]), "t": int(r["st"]["t"])}
        k = start * nper
        log(f"layout AR: resuming after epoch {start} (best val NLL {best:.3f})")
    t0 -= spent
    for ep in range(start, epochs):
        rng = np.random.default_rng(2 + ep)
        perm = rng.permutation(len(Ttr))
        ls = []
        for s in range(0, nper * bs, bs):
            i = perm[s:s + bs]
            T = Ttr[i].copy()
            fl = rng.random(bs) < 0.5
            T[fl] = flip_tokens(T[fl])
            warm = min(1.0, (k + 1) / 300)
            lr = 1e-3 * warm * 0.5 * (1 + np.cos(np.pi * k / steps))
            p, st, l = step(p, st, jnp.asarray(ytr[i]), jnp.asarray(T), lr)
            ls.append(float(l))
            k += 1
        v = np.mean([float(vloss(p, jnp.asarray(yva[s:s + 500]), jnp.asarray(Tva[s:s + 500])))
                     for s in range(0, len(Tva), 500)])
        log(f"layout AR epoch {ep + 1}: train NLL {np.mean(ls):.3f} val NLL {v:.3f} nats/cell "
            f"({v * 64 / np.log(2) / 192:.3f} bits/dim) ({time.time() - t0:.0f}s)")
        if v < best:
            best = v
            nets.save(p, cache("ar.pkl"))
        nets.save({"p": p, "st": st, "best": best, "epoch": ep + 1, "spent": time.time() - t0}, resume)
        if time.time() - t0 > minutes * 60:
            log("layout AR: time budget reached")
            break
    log(f"layout AR: best val NLL {best:.3f}")


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    Xtr, ytr, Xva, yva, src = split()
    log(f"=== fit_models {time.strftime('%Y-%m-%d %H:%M')} data={src} train={len(Xtr)} val={len(Xva)}")
    if only in (None, "palette"):
        fit_palette(Xtr, Xva)
    if only in (None, "closed"):
        fit_closed(Xtr, ytr, Xva)
    if only in (None, "judge"):
        fit_judge(Xtr, ytr, Xva, yva)
    if only in (None, "ar"):
        fit_ar(Xtr, ytr, Xva, yva)
