# PREREG_STOCHASTIC — sample the detail instead of predicting its mean

Written and committed after the main gates (PREREG.md) ran and before any code below existed.

## What the main gates showed

The learned layout is close to the ceiling the detail stage allows: real images reduced to their own
8x8 layout and filled by the Sihti memory sit at FD 102 from real images, the learned layouts at 110,
real-vs-real at 1. Fine-detail energy (sieve r0 + r1) is 11% of real images' even with the true
layout. P3 is a least-squares map per code: it returns the conditional *mean* residue, and the mean
of texture is close to zero.

## The change (arm R4)

Lift + P3 as before, plus a sampled part:

```
residue patch = P3 mean(window) + scale(window) * L_k z,   z ~ N(0, I_16)
```

- `L_k`: Cholesky factor of code k's leftover covariance (16x16, in contrast-normalised units),
  measured on fresh training windows (seed 1, 10 per image per band, the 45 000 training images;
  never the test set). Shrinkage 1e-3 x trace/16 on the diagonal.
- Overlapping patches: the mean part is averaged as before. The sampled part is summed and divided
  by sqrt(count) instead of count, so each pixel keeps the variance of one sample instead of the
  reduced variance of an average of independent samples.
- Bands r2, r1, r0 in that order; each sampled band feeds the next band's windows.
- One fixed random seed for all samples; no temperature, no per-band scale, nothing tuned on any
  measured quantity.

## Gates (same 1 000 learned layouts and same judge as the main run)

| gate | measure | pass if | my prediction |
|---|---|---|---|
| S1 | fine-detail energy of LA+R4 / real | 0.6 - 1.4 | ~0.8 |
| S2 | FD to real test images, LA+R4 vs LA+R3 | R4 < R3 (110.2) | ~95 |
| S3 | PSNR on real layouts, R4 vs R3 | reported only: a cost, expected to fall | about -1.5 dB |

Kill condition: if S2 fails, sampled per-code texture moves samples *away* from real images in the
judge's feature space, and the detail stage stays the unsolved part.

## Post-hoc instrument, reported without a threshold

The main judge (trained on sharp images) reads only 26% of real images filled from their own 8x8
layout, and its predictions on blurry images pile onto bird, cat and ship. A second judge with the
same architecture, trained on bicubic-upsampled 8x8 layouts of the training images (6 epochs), reads
the layout arms in their own domain. This was designed after seeing the main results; it gets no
pass/fail and does not change G1.
