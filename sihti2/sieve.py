"""(Copied unchanged from anttiluode/SihtiImageModel.) The sieve (Sihti, lattice arm).

purify(x) = one heat-diffusion step on the pixel lattice (5-point Laplacian,
reflecting borders, dt = 0.25).  Run it to octave depths d_0 < d_1 < ... and
save every departure

    r_k = x_{d_k} - x_{d_{k+1}}          core = x_{d_last}

so that   x = r_0 + r_1 + ... + core   exactly (telescoping sum).

The first residue r_0 = x - purify(x) = -dt * Laplacian(x) is a centre-surround
(retina-like) band.  Nothing here is learned.
"""
import numpy as np

DEPTHS = (0, 1, 4, 16, 64)          # diffusion steps; 4 residues + core
BAND_NAMES = ("r0 (finest)", "r1", "r2", "r3", "core")


def purify(x, dt=0.25):
    """One diffusion step. x: (..., H, W)."""
    dt = x.dtype.type(dt)
    p = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    lap = (p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2]
           + p[..., 1:-1, 2:] - 4.0 * x)
    return x + dt * lap


def sieve(x, depths=DEPTHS, dtype=np.float64):
    """x: (N, C, H, W) float. Returns list of bands, fine -> coarse, last = core."""
    x = np.asarray(x, dtype=dtype)
    states = []
    cur, step = x, 0
    for d in depths:
        while step < d:
            cur = purify(cur)
            step += 1
        states.append(cur)
    bands = [states[k] - states[k + 1] for k in range(len(states) - 1)]
    bands.append(states[-1])
    return bands


def unsieve(bands):
    """Inverse at the soma: plain addition."""
    out = np.zeros_like(bands[0])
    for b in bands:
        out = out + b
    return out
