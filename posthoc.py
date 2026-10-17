"""PREREG_STOCHASTIC.md: arm R4 (sampled detail) gates S1-S3, and the post-hoc blur-domain judge.

    python posthoc.py          # after gates.py

Writes results/posthoc_results.json, results/posthoc_log.txt, results/cache/posthoc_arrays.npz.
"""
import json
import os
import pickle
import time

import numpy as np

from fit_models import fit_judge, split
from gates import Judge, fd, hf_energy, psnr_each
from sihti2 import nets
from sihti2.data import CLASSES, load
from sihti2.layout import bicubic, chans, layout, refine, to_float
from sihti2.stochastic import fit_residual_chol, refine_stochastic

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(RES, "cache")
LOG = open(os.path.join(RES, "posthoc_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def main():
    t0 = time.time()
    log(f"=== posthoc {time.strftime('%Y-%m-%d %H:%M')}")
    lift = pickle.load(open(os.path.join(CACHE, "lift.pkl"), "rb"))
    pm = pickle.load(open(os.path.join(CACHE, "patch.pkl"), "rb"))
    judge = Judge(nets.load(os.path.join(CACHE, "judge.pkl")))
    z = np.load(os.path.join(CACHE, "gate_arrays.npz"))
    R0 = json.load(open(os.path.join(RES, "results.json")))
    Xtr, ytr, Xva, yva, _ = split()

    path = os.path.join(CACHE, "stoch.pkl")
    if not os.path.exists(path):
        t = time.time()
        pickle.dump(fit_residual_chol(pm, chans(Xtr)), open(path, "wb"))
        log(f"residual covariances fitted in {time.time() - t:.0f}s")
    Lc = pickle.load(open(path, "rb"))

    Xte, yte, _ = load("test")
    Xte = to_float(Xte)
    first = z["first"]
    Xo, yo = Xte[first], yte[first]
    cls = z["cls"]
    f_real = judge.features(Xo)
    hf_real = hf_energy(Xo)
    out, imgs = {}, {}

    # ---------------- S1, S2 on the same learned layouts; LG for reference
    for lname in ("LA", "LG"):
        Y = refine_stochastic(z[f"L_{lname}"], lift, pm, Lc, np.random.default_rng(2027))
        imgs[f"{lname}_R4"] = Y
        out[f"{lname}_R4"] = {"judge_acc": float((judge.predict(Y) == cls).mean()),
                              "fd": fd(judge.features(Y), f_real), "hf_ratio": hf_energy(Y) / hf_real}
        log(f"{lname}+R4 sampled detail: judge {out[f'{lname}_R4']['judge_acc']:.3f}  FD {out[f'{lname}_R4']['fd']:.3f}  "
            f"fine-energy ratio {out[f'{lname}_R4']['hf_ratio']:.3f}")
    hr, f4, f3 = out["LA_R4"]["hf_ratio"], out["LA_R4"]["fd"], R0["generated"]["LA"]["R3"]["fd"]
    S1, S2 = 0.6 <= hr <= 1.4, f4 < f3
    log(f"S1 fine-energy ratio {hr:.3f} -> {'PASS' if S1 else 'FAIL'}")
    log(f"S2 FD R4 {f4:.3f} vs R3 {f3:.3f} -> {'PASS' if S2 else 'FAIL'}")

    # ---------------- S3: the cost on real layouts
    Yo = refine_stochastic(layout(Xo), lift, pm, Lc, np.random.default_rng(2028))
    p4, p3 = float(psnr_each(Yo, Xo).mean()), R0["oracle"]["R3"]["psnr"]
    out["real_R4"] = {"psnr": p4, "judge_acc": float((judge.predict(Yo) == yo).mean()),
                      "fd_to_other_real": None, "hf_ratio": hf_energy(Yo) / hf_real}
    log(f"S3 real layouts: R4 PSNR {p4:.2f} dB vs R3 {p3:.2f} dB ({p4 - p3:+.2f}); fine-energy {out['real_R4']['hf_ratio']:.3f}; "
        f"judge {out['real_R4']['judge_acc']:.3f}")
    Xo2 = Xte[np.concatenate([np.where(yte == c)[0][200:400] for c in range(10)])]
    out["real_R4"]["fd_to_other_real"] = fd(judge.features(Yo), judge.features(Xo2))
    log(f"S3 real layouts: R4 FD to other real images {out['real_R4']['fd_to_other_real']:.3f} "
        f"(R3: {R0['generated']['real_layout']['R3']['fd']:.3f})")

    # ---------------- post-hoc blur-domain judge
    bj_path = os.path.join(CACHE, "judge_blur.pkl")
    if not os.path.exists(bj_path):
        t = time.time()
        fit_judge(bicubic(layout(Xtr)), ytr, bicubic(layout(Xva)), yva, epochs=6, minutes=30,
                  out="judge_blur.pkl", log_fn=log, tag="blur judge")
        log(f"blur judge trained in {time.time() - t:.0f}s")
    bj = Judge(nets.load(bj_path))
    arms = {"real layouts, bicubic": (bicubic(layout(Xo)), yo),
            "real layouts, Sihti P3": (refine(layout(Xo), lift, pm, "R3"), yo),
            "real images (sharp)": (Xo, yo)}
    for lname in ("LA", "LG"):
        for a in ("R1", "R3"):
            arms[f"{lname}+{a}"] = (z[f"{lname}_{a}"].astype(np.float32), cls)
        arms[f"{lname}+R4"] = (imgs[f"{lname}_R4"], cls)
    bjr = {}
    for name, (X, y) in arms.items():
        pr = bj.predict(X)
        bjr[name] = {"acc": float((pr == y).mean()),
                     "per_class": {CLASSES[c]: float((pr[y == c] == c).mean()) for c in range(10)},
                     "predicted_share": {CLASSES[c]: float((pr == c).mean()) for c in range(10)}}
        log(f"blur judge  {name:24s} {bjr[name]['acc']:.3f}")
    out["blur_judge"] = bjr
    out["S1"], out["S2"] = S1, S2
    json.dump(out, open(os.path.join(RES, "posthoc_results.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(CACHE, "posthoc_arrays.npz"), LA_R4=imgs["LA_R4"].astype(np.float16),
                        LG_R4=imgs["LG_R4"].astype(np.float16), real_R4=Yo.astype(np.float16))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
