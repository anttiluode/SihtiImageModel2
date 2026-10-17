"""CIFAR-10 loader. Returns uint8 images (N, 32, 32, 3) and int labels.

Preference order:
  1. data/cifar-10-batches-py/   the official lossless batches (Toronto). Downloaded on first use
     with `python -m sihti2.data --official` if your machine has normal internet.
  2. data/cifar_png/             the JPEG mirror github.com/YoongiKim/CIFAR-10-images
     (same 50 000 / 10 000 split, but re-encoded as JPEG at quality ~75 with 4:2:0 chroma).
     The runs reported in README.md used this mirror, because the build machine could not reach
     the Toronto site. JPEG blurs the finest band and adds 8x8 block edges, so r0 numbers on the
     official data will differ.
A cache data/cifar10_<source>.npz is written on first load.
"""
import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def _official():
    d = os.path.join(ROOT, "cifar-10-batches-py")
    if not os.path.isdir(d):
        return None
    out = {}
    for split, names in (("train", [f"data_batch_{i}" for i in range(1, 6)]), ("test", ["test_batch"])):
        Xs, ys = [], []
        for n in names:
            with open(os.path.join(d, n), "rb") as f:
                b = pickle.load(f, encoding="bytes")
            Xs.append(b[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys.append(np.array(b[b"labels"]))
        out[split] = (np.concatenate(Xs).astype(np.uint8), np.concatenate(ys).astype(np.int64))
    return out


def _mirror():
    from PIL import Image
    d = os.path.join(ROOT, "cifar_png")
    if not os.path.isdir(d):
        return None
    out = {}
    for split in ("train", "test"):
        Xs, ys = [], []
        for c, name in enumerate(CLASSES):
            fd = os.path.join(d, split, name)
            for fn in sorted(os.listdir(fd)):
                Xs.append(np.asarray(Image.open(os.path.join(fd, fn)).convert("RGB")))
                ys.append(c)
        out[split] = (np.stack(Xs).astype(np.uint8), np.array(ys, np.int64))
    return out


def load(split="train"):
    """Returns X uint8 (N,32,32,3), y int64, source name."""
    for src, fn in (("official", _official), ("jpeg_mirror", _mirror)):
        cache = os.path.join(ROOT, f"cifar10_{src}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            return z[f"X_{split}"], z[f"y_{split}"], src
        data = fn()
        if data is not None:
            np.savez(cache, X_train=data["train"][0], y_train=data["train"][1],
                     X_test=data["test"][0], y_test=data["test"][1])
            return data[split][0], data[split][1], src
    raise FileNotFoundError("No CIFAR-10 found. Run `python -m sihti2.data --official` (needs internet), or\n"
                            "git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images data/cifar_png")


if __name__ == "__main__":
    if "--official" in sys.argv:
        os.makedirs(ROOT, exist_ok=True)
        tgz = os.path.join(ROOT, "cifar-10-python.tar.gz")
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(ROOT)
    X, y, src = load("train")
    print(src, X.shape, np.bincount(y))
