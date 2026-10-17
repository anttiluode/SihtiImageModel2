"""Pre-registered gates (PREREG.md). Run after fit_models.py.

    python gates.py

Writes results/results.json, results/gate_log.txt and results/cache/gate_arrays.npz (for figures/web).
"""
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg

from sihti2 import nets
from sihti2.data import CLASSES, load
from sihti2.layout import chans, layout, refine, to_float, unchans, x16
from sihti2.patch import bands_of

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(RES, "cache")
ARMS = ["R0", "R1", "R2", "R3"]
ARM_NAMES = {"R0": "lift only", "R1": "bicubic", "R2": "lift+P0 linear", "R3": "lift+P3 Sihti memory"}
LOG = open(os.path.join(RES, "gate_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def psnr_each(a, b):
    mse = ((np.clip(a, 0, 1) - b) ** 2).reshape(len(a), -1).mean(1)
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


class Judge:
    def __init__(self, p):
        self.p = p
        self._f = jax.jit(nets.cnn_features)
        self._l = jax.jit(nets.cnn_logits)

    def _run(self, f, X):
        X = np.clip(X, 0, 1).astype(np.float32)
        return np.concatenate([np.asarray(f(self.p, jnp.asarray(X[s:s + 500]))) for s in range(0, len(X), 500)])

    def predict(self, X):
        return self._run(self._l, X).argmax(1)

    def features(self, X):
        return self._run(self._f, X).astype(np.float64)


def fd(fa, fb):
    m1, m2 = fa.mean(0), fb.mean(0)
    c1, c2 = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    s = linalg.sqrtm(c1 @ c2)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * s.real))


def hf_energy(X):
    B = bands_of(chans(np.clip(X, 0, 1)))
    return float(((B[0] + B[1]) ** 2).mean())


def nn_dist(Q, R, chunk=2000):
    Q = Q.reshape(len(Q), -1).astype(np.float32)
    R = R.reshape(len(R), -1).astype(np.float32)
    r2 = (R ** 2).sum(1)
    out, idx = np.empty(len(Q)), np.empty(len(Q), np.int64)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d2 = (q ** 2).sum(1)[:, None] + r2[None] - 2 * q @ R.T
        idx[s:s + chunk] = d2.argmin(1)
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out, idx


def main():
    t0 = time.time()
    lift = pickle.load(open(os.path.join(CACHE, "lift.pkl"), "rb"))
    pm = pickle.load(open(os.path.join(CACHE, "patch.pkl"), "rb"))
    gauss = pickle.load(open(os.path.join(CACHE, "gauss.pkl"), "rb"))
    pal = pickle.load(open(os.path.join(CACHE, "palette.pkl"), "rb"))
    judge = Judge(nets.load(os.path.join(CACHE, "judge.pkl")))
    ar = nets.load(os.path.join(CACHE, "ar.pkl"))

    Xte, yte, src = load("test")
    Xte = to_float(Xte)
    Xtr_u8, ytr, _ = load("train")
    Ltr = layout(to_float(Xtr_u8))
    first = np.concatenate([np.where(yte == c)[0][:200] for c in range(10)])
    second = np.concatenate([np.where(yte == c)[0][200:400] for c in range(10)])
    Xo, yo, Xo2 = Xte[first], yte[first], Xte[second]
    log(f"=== gates {time.strftime('%Y-%m-%d %H:%M')} data={src}")
    R = {"data": src, "arms": ARM_NAMES}

    judge_full = float((judge.predict(Xte) == yte).mean())
    judge_o = float((judge.predict(Xo) == yo).mean())
    log(f"judge on real test images: all 10000 {judge_full:.4f}, gate subset {judge_o:.4f}")
    R["judge_real_test"], R["judge_real_subset"] = judge_full, judge_o

    # ------------------------------------------------------------ O1, O2: real layouts
    Lo = layout(Xo)
    outs = {a: refine(Lo, lift, pm, a) for a in ARMS}
    ps = {a: psnr_each(outs[a], Xo) for a in ARMS}
    # reference only: P3 fill from the TRUE x16 (what v1 called "core + r3")
    cur = chans(x16(Xo))
    for b in (2, 1, 0):
        cur = cur + pm.predict_band(cur, b, "P3")
    ref = unchans(cur, len(Xo))
    o = {a: {"psnr": float(ps[a].mean()), "judge_acc": float((judge.predict(outs[a]) == yo).mean())} for a in ARMS}
    o["P3_from_true_x16_reference"] = {"psnr": float(psnr_each(ref, Xo).mean()),
                                       "judge_acc": float((judge.predict(ref) == yo).mean())}
    o["lift_x16_psnr_vs_true_x16"] = float(psnr_each(lift(Lo), x16(Xo)).mean())
    o["R3_beats_R2_on_fraction"] = float((ps["R3"] > ps["R2"]).mean())
    o["R3_beats_R1_on_fraction"] = float((ps["R3"] > ps["R1"]).mean())
    for a in ARMS:
        log(f"O real layouts {a:3s} {ARM_NAMES[a]:22s} PSNR {o[a]['psnr']:.2f} dB  judge {o[a]['judge_acc']:.4f}")
    log(f"O reference: P3 from true x16: {o['P3_from_true_x16_reference']}")
    log(f"O R3 > R2 on {o['R3_beats_R2_on_fraction']:.3f} of images, R3 > R1 on {o['R3_beats_R1_on_fraction']:.3f}")
    d32, d31 = o["R3"]["psnr"] - o["R2"]["psnr"], o["R3"]["psnr"] - o["R1"]["psnr"]
    O1 = d32 >= 0.1 and d31 > 0
    O2 = o["R3"]["judge_acc"] >= o["R1"]["judge_acc"] + 0.02
    log(f"O1 R3-R2 {d32:+.3f} dB, R3-R1 {d31:+.3f} dB -> {'PASS' if O1 else 'FAIL'}")
    log(f"O2 judge R3 {o['R3']['judge_acc']:.4f} vs R1 {o['R1']['judge_acc']:.4f} -> {'PASS' if O2 else 'FAIL'}")
    R["oracle"], R["O1"], R["O2"] = o, O1, O2

    # ------------------------------------------------------------ generation
    cls = np.repeat(np.arange(10), 100)
    t = time.time()
    T = nets.ar_sample(ar, jnp.asarray(cls), jax.random.PRNGKey(2026), 1.0)
    L_LA = pal.decode(T)
    log(f"sampled 1000 LA layouts in {time.time() - t:.0f}s")
    L_LG = gauss.sample(cls, np.random.default_rng(2026))
    f_real = judge.features(Xo)
    f_real2 = judge.features(Xo2)
    R["fd_real_vs_real"] = fd(f_real, f_real2)
    hf_real = hf_energy(Xo)
    log(f"FD real 2000 vs other real 2000 (floor): {R['fd_real_vs_real']:.3f}; real fine-band energy {hf_real:.3e}")
    gen, gen_imgs = {}, {}
    for lname, L in (("LA", L_LA), ("LG", L_LG)):
        gen[lname] = {}
        for a in ARMS:
            Y = refine(L, lift, pm, a)
            gen_imgs[f"{lname}_{a}"] = Y
            g = {"judge_acc": float((judge.predict(Y) == cls).mean()),
                 "fd": fd(judge.features(Y), f_real),
                 "hf_ratio": hf_energy(Y) / hf_real}
            gen[lname][a] = g
            log(f"G {lname}+{a} {ARM_NAMES[a]:22s} judge {g['judge_acc']:.3f}  FD {g['fd']:.3f}  fine-energy ratio {g['hf_ratio']:.3f}")
    # real images through their own layout with each arm, for FD reference
    gen["real_layout"] = {a: {"fd": fd(judge.features(outs[a]), f_real2), "hf_ratio": hf_energy(outs[a]) / hf_real} for a in ARMS}
    R["generated"] = gen

    la3, lg3 = gen["LA"]["R3"]["judge_acc"], gen["LG"]["R3"]["judge_acc"]
    G1 = la3 >= 0.35 and la3 >= lg3 + 0.10
    f = {a: gen["LA"][a]["fd"] for a in ARMS}
    G2 = f["R3"] < f["R2"] < f["R1"] and f["R3"] < f["R0"]
    hr = gen["LA"]["R3"]["hf_ratio"]
    G2b = 0.6 <= hr <= 1.4
    log(f"G1 LA+R3 {la3:.3f} vs LG+R3 {lg3:.3f} -> {'PASS' if G1 else 'FAIL'}")
    log(f"G2 FD R3 {f['R3']:.3f} R2 {f['R2']:.3f} R1 {f['R1']:.3f} R0 {f['R0']:.3f} -> {'PASS' if G2 else 'FAIL'}")
    log(f"G2b fine-energy ratio LA+R3 {hr:.3f} -> {'PASS' if G2b else 'FAIL'}")

    # ------------------------------------------------------------ G3 novelty
    d_test, _ = nn_dist(Lo, Ltr)
    med, p1 = np.median(d_test), np.percentile(d_test, 1)
    nov = {}
    for lname, L in (("LA", L_LA), ("LG", L_LG)):
        d, idx = nn_dist(L, Ltr)
        nov[lname] = {"ratio": float(np.median(d) / med), "near_copies": float((d < p1).mean())}
        if lname == "LA":
            nn_idx = idx
        log(f"G3 {lname}: novelty ratio {nov[lname]['ratio']:.3f}, near-copies {nov[lname]['near_copies']:.4f}")
    G3 = nov["LA"]["ratio"] >= 0.7 and nov["LA"]["near_copies"] <= 0.02
    log(f"G3 -> {'PASS' if G3 else 'FAIL'}")
    R["novelty"], R["G1"], R["G2"], R["G2b"], R["G3"] = nov, G1, G2, G2b, G3

    # per-class recognition for LA+R3
    pr = judge.predict(gen_imgs["LA_R3"])
    R["LA_R3_per_class"] = {CLASSES[c]: float((pr[cls == c] == c).mean()) for c in range(10)}
    log("LA+R3 per class: " + ", ".join(f"{k} {v:.2f}" for k, v in R["LA_R3_per_class"].items()))

    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(CACHE, "gate_arrays.npz"), cls=cls, T=T, L_LA=L_LA, L_LG=L_LG,
                        nn_idx=nn_idx, first=first, **{k: v.astype(np.float16) for k, v in gen_imgs.items()},
                        **{f"oracle_{a}": outs[a].astype(np.float16) for a in ARMS})
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
