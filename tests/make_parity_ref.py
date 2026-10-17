"""Reference outputs from the Python model for tests/test_web_parity.mjs.

    python tests/make_parity_ref.py && node tests/test_web_parity.mjs
"""
import json
import os
import pickle
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sihti2 import nets
from sihti2.layout import bicubic, refine

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "results", "cache")

pal = pickle.load(open(os.path.join(CACHE, "palette.pkl"), "rb"))
lift = pickle.load(open(os.path.join(CACHE, "lift.pkl"), "rb"))
pm = pickle.load(open(os.path.join(CACHE, "patch.pkl"), "rb"))
ar = nets.load(os.path.join(CACHE, "ar.pkl"))
rng = np.random.default_rng(7)
toks = rng.integers(0, pal.K, 64).astype(np.int32)
lg = np.asarray(nets.ar_logits(ar, jnp.array([2]), jnp.asarray(toks[None])))[0]
L = pal.decode(toks[None])
ref = {"cls": 2, "toks": toks.tolist(), "logits": lg[[0, 1, 30, 63]].tolist(), "logit_pos": [0, 1, 30, 63],
       "layout": L.reshape(-1).tolist(), "lift": lift(L).reshape(-1).tolist(),
       "p3": refine(L, lift, pm, "R3").reshape(-1).tolist(), "p0": refine(L, lift, pm, "R2").reshape(-1).tolist(),
       "bicubic": bicubic(L).reshape(-1).tolist()}
json.dump(ref, open(os.path.join(HERE, "tests", "parity_ref.json"), "w"))
print("wrote tests/parity_ref.json")
