"""Export the fitted model for the browser page (web/).

    python export_web.py

web/model.bin   all arrays as float16, concatenated
web/model.json  manifest {name: [offset, shape]} + bench images + results
The page (web/model.js) re-implements nets.ar_logits, Lift and PatchMemory.predict_band(P3)
and is checked against Python by tests/test_web_parity.mjs.
"""
import json
import os
import pickle

import numpy as np

from sihti2 import nets
from sihti2.data import CLASSES, load
from sihti2.layout import to_float

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "results", "cache")
WEB = os.path.join(HERE, "web")


def main():
    pal = pickle.load(open(os.path.join(CACHE, "palette.pkl"), "rb"))
    lift = pickle.load(open(os.path.join(CACHE, "lift.pkl"), "rb"))
    pm = pickle.load(open(os.path.join(CACHE, "patch.pkl"), "rb"))
    ar = nets.load(os.path.join(CACHE, "ar.pkl"))
    arrays = {"palette": pal.C, "lift": lift.W}
    for k in ("tok", "cls", "pos", "lnf_g", "lnf_b", "head"):
        arrays[f"ar.{k}"] = np.asarray(ar[k])
    for i, b in enumerate(ar["blocks"]):
        for k, v in b.items():
            arrays[f"ar.{i}.{k}"] = np.asarray(v)
    for b in (2, 1, 0):
        arrays[f"p3.{b}.C"] = pm.arms[b]["C"]
        arrays[f"p3.{b}.W"] = pm.arms[b]["W"]
        arrays[f"p0.{b}.W"] = pm.arms[b]["W0"]
    stoch = pickle.load(open(os.path.join(CACHE, "stoch.pkl"), "rb"))
    for b in (2, 1, 0):
        arrays[f"p3s.{b}.L"] = stoch[b]
    man, off, chunks = {}, 0, []
    for k, v in arrays.items():
        a = np.ascontiguousarray(np.asarray(v, np.float32)).astype(np.float16)
        man[k] = [off, list(a.shape)]
        chunks.append(a.tobytes())
        off += a.size
    open(os.path.join(WEB, "model.bin"), "wb").write(b"".join(chunks))
    # bench: 4 held-out test images per class, never used in fitting
    X, y, src = load("test")
    idx = np.concatenate([np.where(y == c)[0][300:304] for c in range(10)])
    res = json.load(open(os.path.join(HERE, "results", "results.json")))
    ph = os.path.join(HERE, "results", "posthoc_results.json")
    if os.path.exists(ph):
        res["posthoc"] = json.load(open(ph))
    meta = {"arrays": man, "classes": CLASSES, "n_layers": len(ar["blocks"]), "heads": 4,
            "bench": {"images": X[idx].reshape(len(idx), -1).tolist(), "labels": y[idx].tolist()},
            "data": src, "results": res}
    json.dump(meta, open(os.path.join(WEB, "model.json"), "w"))
    import shutil
    for f in ("fig_samples.png", "fig_stages.png", "fig_nearest.png", "fig_real_layouts.png", "fig_sampled_detail.png"):
        shutil.copy(os.path.join(HERE, "results", f), os.path.join(WEB, f))
    print(f"model.bin {off * 2 / 1e6:.2f} MB, {len(man)} arrays")


if __name__ == "__main__":
    main()
