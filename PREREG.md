# PREREG — SihtiImageModel2: a learned layout, Sihti detail

Committed before any gate below was run. Thresholds and predictions are not moved afterwards; any
change is logged in the README as a correction with both numbers.

## Why this repo exists

SihtiImageModel (v1) generated recognisable digits and clothing with no backprop, and failed on
CIFAR-10: the whole-image memory returned colour layouts, not objects, and its patch memory could
only fill detail *below* a layout it was given. v1's own conclusion: generating a photo needs a
global model of the layout; the patch memory fills in below it.

This repo tests that split directly:

```
class -> layout model -> 8x8x3 layout L      (the global part; learned, backprop allowed HERE ONLY)
L -> lift -> x16                              (closed form: ridge map to Sihti's coarse state core+r3)
x16 -> patch memory -> r2, r1, r0             (closed form: v1's P3 codebook of local linear maps)
image = x16 + r2 + r1 + r0
```

The two questions are separate and get separate gates:

1. **Layout.** Does a small learned layout model give object-like layouts that a closed-form
   Gaussian layout model does not?
2. **Detail.** Under a layout (real or generated), does the Sihti patch memory add anything over
   ordinary upsampling and over a single learned linear filter?

## Data

CIFAR-10, 50 000 train / 10 000 test. On the build machine only the JPEG mirror
github.com/YoongiKim/CIFAR-10-images was reachable (quality ~75, 4:2:0 chroma); the README states
which source each run used. 5 000 train images (500 per class, the last 500 of each class) are held
out for the layout model's validation. The test set is touched only by the gates.

## Fixed design

- **Layout** L = 4x4 average pool of the image, 8x8x3 = 192 numbers in [0, 1].
- **Lift** one ridge map (192 -> 3072) from L to x16 = core + r3 of v1's lattice sieve
  (diffusion depths 0, 1, 4, 16, 64), fitted on train.
- **Detail arms**, all applied per colour channel:
  - R0 lift only (no detail)
  - R1 bicubic upsampling of L (PIL), the ordinary baseline
  - R2 lift + v1's P0 (one learned linear filter per band)
  - **R3 lift + v1's P3 (512 codes per band, each a local linear map; main arm)**
  P0/P3 code is copied unchanged from v1 (`sihti2/patch.py`), fitted on train images with all
  three channels pooled as grey images (10 windows per image per band), and asked to fill r2, r1, r0
  in that order from the lifted x16.
- **Layout arms**
  - LG: per-class Gaussian (mean + covariance of the 192 numbers, shrinkage 1e-3), no backprop
  - **LA: class-conditional autoregressive transformer** over the 64 cells in raster order. Each
    cell's colour is quantised to a 512-colour palette (k-means on train cells). 4 layers, width
    128, 4 heads, class token first. Adam, horizontal-flip augmentation, at most 15 epochs or
    75 minutes of training, checkpoint chosen by validation NLL. Sampling at temperature 1.0 from
    the full distribution (no top-k, no truncation).
  - LO: oracle, the real layout of a test image (for the detail gates)
- **Judge** a small CNN (32-32-pool-64-64-pool-128-global-pool-10), trained with backprop on the
  full-resolution train images, flip augmentation. Its test accuracy on real images is reported as
  the ceiling. Its 128-d pooled features are used for a Frechet distance (FD) between image sets.
  It is not an Inception network, so FD here is not comparable to published FID numbers.

## Gates

Real layouts (LO): 2 000 test images, the first 200 of each class.

| gate | measure | pass if | my prediction |
|---|---|---|---|
| O1 | PSNR to the true image | R3 beats R2 by >= 0.1 dB on the mean AND beats R1 (bicubic) | R3 - R1 about +1.0 dB, R3 - R2 about +0.2 dB |
| O2 | judge accuracy on the reconstructions | R3 >= R1 + 2 points | R1 ~45%, R3 ~50%, real ~78% |

Generated (100 samples per class per layout arm, 1 000 each):

| gate | measure | pass if | my prediction |
|---|---|---|---|
| G1 | judge reads the requested class (LA+R3) | >= 35% AND >= LG+R3 + 10 points | LA 40%, LG 20% |
| G2 | FD to 2 000 real test images, LA layouts | R3 < R2 < R1 and R3 < R0 | holds |
| G2b | fine-detail energy (sieve r0 + r1) of LA+R3 samples / of real test images | within 0.6 - 1.4 | ~0.8; bicubic ~0.3 |
| G3 | novelty: median (sample layout -> nearest train layout) / (test layout -> nearest train layout) | >= 0.7, and <= 2% near-copies (closer than the 1st percentile of test -> train) | ~0.9, 0 copies |

## Kill conditions

- **G1 fails on the comparison with LG**: the learned layout buys nothing over a Gaussian; the
  "layout is the missing part" story is wrong at this scale.
- **O1 and G2 both fail for R3**: the Sihti detail stage adds nothing a linear filter or bicubic
  upsampling does not, once a layout exists. Then the honest system is "learned layout + ordinary
  upsampling" and the residue memory is not earning its place.

## Not claimed in advance

Samples that look like CIFAR photographs. 8x8 is a very small layout; the expectation is
recognisable object-shaped colour arrangements with plausible edges, at the level of early
autoregressive CIFAR samples, not modern diffusion samples.
