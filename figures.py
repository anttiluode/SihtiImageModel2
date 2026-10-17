"""Figures from results/cache/gate_arrays.npz (run gates.py first).  python figures.py"""
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sihti2.data import CLASSES, load
from sihti2.layout import refine, to_float

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CACHE = os.path.join(RES, "cache")


def show(ax, im, title=None):
    ax.imshow(np.clip(np.asarray(im, np.float32), 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)


def grid(ax, ims, rows, cols, title):
    H = W = ims.shape[1]
    canvas = np.ones((rows * (H + 2), cols * (W + 2), 3), np.float32)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        canvas[r * (H + 2):r * (H + 2) + H, c * (W + 2):c * (W + 2) + W] = np.clip(ims[k], 0, 1)
    show(ax, canvas, title)


def main():
    z = np.load(os.path.join(CACHE, "gate_arrays.npz"))
    cls = z["cls"]
    pick = np.concatenate([np.where(cls == c)[0][:8] for c in range(10)])

    # 1. samples: three panels on the same class rows
    fig, axs = plt.subplots(1, 3, figsize=(18, 7.6))
    grid(axs[0], z["LA_R3"][pick].astype(np.float32), 10, 8, "learned layout + Sihti patch memory (LA+R3)")
    grid(axs[1], z["LA_R1"][pick].astype(np.float32), 10, 8, "same layouts, bicubic upsampling (LA+R1)")
    grid(axs[2], z["LG_R3"][pick].astype(np.float32), 10, 8, "Gaussian layout + Sihti patch memory (LG+R3)")
    for k, n in enumerate(CLASSES):
        axs[0].text(-3, k * 34 + 17, n, ha="right", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_samples.png"), dpi=110)
    plt.close()

    # 2. stages for one sample per class
    lift = pickle.load(open(os.path.join(CACHE, "lift.pkl"), "rb"))
    pm = pickle.load(open(os.path.join(CACHE, "patch.pkl"), "rb"))
    sel = np.array([np.where(cls == c)[0][0] for c in range(10)])
    L = z["L_LA"][sel]
    out, stages, bands = refine(L, lift, pm, "R3", return_stages=True)
    bic = refine(L, lift, pm, "R1")
    cols = ["8x8 layout", "lift (x16)", "+ r2", "+ r1", "+ r0 = image", "r2", "r1", "r0", "bicubic"]
    fig, axs = plt.subplots(10, len(cols), figsize=(len(cols) * 1.35, 10 * 1.4))
    for i in range(10):
        ims = [np.kron(L[i], np.ones((4, 4, 1)))] + stages + [None, None, None, bic[i:i + 1]]
        for j in range(len(cols)):
            if 5 <= j <= 7:
                b = bands[j - 5][i]
                s = np.abs(b).max() + 1e-9
                show(axs[i, j], 0.5 + 0.5 * b / s)
            else:
                show(axs[i, j], ims[j][i] if j in (1, 2, 3, 4) else ims[j][0] if j == 8 else ims[0])
            if i == 0:
                axs[i, j].set_title(cols[j], fontsize=8)
        axs[i, 0].set_ylabel(CLASSES[i], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_stages.png"), dpi=110)
    plt.close()

    # 3. real layouts
    Xte, yte, _ = load("test")
    first = z["first"]
    k = np.concatenate([np.arange(c * 200, c * 200 + 2) for c in range(10)])
    truth = to_float(Xte[first[k]])
    cols = [("truth", truth), ("its 8x8 layout", np.kron(truth.reshape(-1, 8, 4, 8, 4, 3).mean((2, 4)), np.ones((1, 4, 4, 1)))),
            ("bicubic (R1)", z["oracle_R1"][k]), ("lift+P0 (R2)", z["oracle_R2"][k]), ("lift+P3 (R3)", z["oracle_R3"][k])]
    fig, axs = plt.subplots(len(cols), 1, figsize=(16, 5.2))
    for j, (n, ims) in enumerate(cols):
        grid(axs[j], np.asarray(ims, np.float32), 1, len(k), None)
        axs[j].set_ylabel(n, fontsize=8, rotation=0, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_real_layouts.png"), dpi=110)
    plt.close()

    # 4. samples beside nearest training image (by layout)
    Xtr, _, _ = load("train")
    sel = np.concatenate([np.where(cls == c)[0][:3] for c in range(10)])
    pairs = []
    for i in sel:
        pairs += [z["LA_R3"][i].astype(np.float32), to_float(Xtr[z["nn_idx"][i]])]
    fig, ax = plt.subplots(figsize=(16, 3.4))
    grid(ax, np.stack(pairs), 3, 20, "pairs: LA+R3 sample | nearest training image by layout")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_nearest.png"), dpi=110)
    plt.close()
    print("figures written")


if __name__ == "__main__":
    main()
