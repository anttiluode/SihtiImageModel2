// SihtiImageModel2 in the browser: the layout transformer, the lift and the Sihti patch memory.
// A line-by-line port of sihti2/nets.py (ar_logits), sihti2/layout.py (Lift, refine) and
// sihti2/patch.py (PatchMemory.predict_band). tests/test_web_parity.mjs checks it against Python.

const STRIDES = { 2: 2, 1: 1, 0: 1 };
const G = 5, P = 4, EPS = 0.02;
const GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]; // _grid(32) in patch.py

function f16(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 5.960464477539063e-8
      : e === 31 ? (m ? NaN : s * Infinity)
      : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}

export function loadModel(meta, buffer) {
  const all = f16(new Uint16Array(buffer));
  const A = {};
  for (const [k, [off, shape]] of Object.entries(meta.arrays)) {
    const n = shape.reduce((a, b) => a * b, 1);
    A[k] = { d: all.subarray(off, off + n), shape };
  }
  return { A, meta, d: A["ar.pos"].shape[1], K: A["palette"].shape[0], L: meta.n_layers, H: meta.heads };
}

// ---------------------------------------------------------------- layout transformer
function matvec(x, W, nIn, nOut, bias) {
  const y = new Float32Array(nOut);
  if (bias) y.set(bias);
  for (let i = 0; i < nIn; i++) {
    const xi = x[i]; if (xi === 0) continue;
    const row = i * nOut;
    for (let j = 0; j < nOut; j++) y[j] += xi * W[row + j];
  }
  return y;
}
function layerNorm(x, g, b) {
  const n = x.length; let mu = 0, v = 0;
  for (let i = 0; i < n; i++) mu += x[i];
  mu /= n;
  for (let i = 0; i < n; i++) v += (x[i] - mu) ** 2;
  const r = 1 / Math.sqrt(v / n + 1e-5), y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mu) * r * g[i] + b[i];
  return y;
}
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

export function newSequence(M) {
  const d = M.d;
  return { pos: 0, toks: [], kc: Array.from({ length: M.L }, () => new Float32Array(64 * d)),
           vc: Array.from({ length: M.L }, () => new Float32Array(64 * d)) };
}

// logits for the next token, given the class (position 0) or the previous token
export function stepLogits(M, seq, cls) {
  const { A, d, L, H } = M, hd = d / H, i = seq.pos;
  let x = new Float32Array(d);
  const src = i === 0 ? A["ar.cls"].d.subarray(cls * d, cls * d + d)
                      : A["ar.tok"].d.subarray(seq.toks[i - 1] * d, seq.toks[i - 1] * d + d);
  const pe = A["ar.pos"].d.subarray(i * d, i * d + d);
  for (let j = 0; j < d; j++) x[j] = src[j] + pe[j];
  for (let l = 0; l < L; l++) {
    const w = (k) => A[`ar.${l}.${k}`].d;
    const h = layerNorm(x, w("ln1_g"), w("ln1_b"));
    const qkv = matvec(h, w("qkv"), d, 3 * d, w("qkv_b"));
    seq.kc[l].set(qkv.subarray(d, 2 * d), i * d);
    seq.vc[l].set(qkv.subarray(2 * d, 3 * d), i * d);
    const y = new Float32Array(d);
    for (let hh = 0; hh < H; hh++) {
      const o = hh * hd, sc = new Float32Array(i + 1);
      let mx = -Infinity;
      for (let t = 0; t <= i; t++) {
        let s = 0;
        for (let e = 0; e < hd; e++) s += qkv[o + e] * seq.kc[l][t * d + o + e];
        sc[t] = s / Math.sqrt(hd); if (sc[t] > mx) mx = sc[t];
      }
      let z = 0;
      for (let t = 0; t <= i; t++) { sc[t] = Math.exp(sc[t] - mx); z += sc[t]; }
      for (let t = 0; t <= i; t++) {
        const a = sc[t] / z;
        for (let e = 0; e < hd; e++) y[o + e] += a * seq.vc[l][t * d + o + e];
      }
    }
    const att = matvec(y, w("o"), d, d, w("o_b"));
    for (let j = 0; j < d; j++) x[j] += att[j];
    const h2 = layerNorm(x, w("ln2_g"), w("ln2_b"));
    const m = matvec(h2, w("w1"), d, 4 * d, w("b1"));
    for (let j = 0; j < m.length; j++) m[j] = gelu(m[j]);
    const m2 = matvec(m, w("w2"), 4 * d, d, w("b2"));
    for (let j = 0; j < d; j++) x[j] += m2[j];
  }
  x = layerNorm(x, A["ar.lnf_g"].d, A["ar.lnf_b"].d);
  return matvec(x, A["ar.head"].d, d, M.K);
}

export function sampleNext(M, seq, cls, rand, temperature = 1) {
  const lg = stepLogits(M, seq, cls);
  let mx = -Infinity;
  for (const v of lg) if (v > mx) mx = v;
  let z = 0;
  const p = lg.map((v) => { const e = Math.exp((v - mx) / temperature); z += e; return e; });
  let r = rand() * z, k = 0;
  for (; k < p.length - 1; k++) { r -= p[k]; if (r <= 0) break; }
  seq.toks.push(k); seq.pos++;
  return k;
}

export function tokensToLayout(M, toks) {
  const L = new Float32Array(192), C = M.A["palette"].d;
  toks.forEach((t, i) => { for (let c = 0; c < 3; c++) L[i * 3 + c] = C[t * 3 + c]; });
  return L;
}

// ---------------------------------------------------------------- lift and patch memory
export function liftLayout(M, L) {         // L: 192 (8,8,3) -> 3072 (32,32,3)
  const W = M.A["lift"].d, x = new Float32Array(193);
  x.set(L); x[192] = 1;
  return matvec(x, W, 193, 3072);
}

function gauss(rand) {
  let u = 0; while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// arm "P3" = mean residue (least squares); arm "P3s" = mean + sample of the code's leftover residue
function predictBand(M, C, b, arm, rand) {        // C: one channel, 32*32
  const sample = arm === "P3s";
  if (sample) arm = "P3";
  const s = STRIDES[b], out = new Float32Array(1024), cnt = new Float32Array(1024), noise = new Float32Array(1024);
  const Lc = sample ? M.A[`p3s.${b}.L`].d : null, z = new Float32Array(16);
  const Cb = arm === "P3" ? M.A[`p3.${b}.C`].d : null;
  const Wb = arm === "P3" ? M.A[`p3.${b}.W`].d : M.A[`p0.${b}.W`].d;
  const K = arm === "P3" ? M.A[`p3.${b}.C`].shape[0] : 0;
  let c2 = null;
  if (Cb) { c2 = new Float32Array(K); for (let k = 0; k < K; k++) { let t = 0; for (let e = 0; e < 25; e++) t += Cb[k * 25 + e] ** 2; c2[k] = t; } }
  const a = new Float32Array(26);
  for (const i of GRID) for (const j of GRID) {
    let n = 0;
    for (let r = -2; r <= 2; r++) for (let q = -2; q <= 2; q++) {
      const y = Math.min(31, Math.max(0, i + 2 + r * s)), x = Math.min(31, Math.max(0, j + 2 + q * s));
      a[n++] = C[y * 32 + x];
    }
    a[25] = 1;
    let W = Wb, sd = 1, best = 0;
    if (arm === "P3") {
      let mu = 0; for (let e = 0; e < 25; e++) mu += a[e];
      mu /= 25;
      let v = 0; for (let e = 0; e < 25; e++) v += (a[e] - mu) ** 2;
      sd = Math.sqrt(v / 25) + EPS;
      for (let e = 0; e < 25; e++) a[e] = (a[e] - mu) / sd;
      let bd = Infinity;
      for (let k = 0; k < K; k++) {
        let dot = 0; for (let e = 0; e < 25; e++) dot += a[e] * Cb[k * 25 + e];
        const dd = c2[k] - 2 * dot; if (dd < bd) { bd = dd; best = k; }
      }
      W = Wb.subarray(best * 26 * 16, (best + 1) * 26 * 16);
    }
    const v = matvec(a, W, 26, 16);
    let nv = null;
    if (sample) {
      for (let e = 0; e < 16; e++) z[e] = gauss(rand);
      nv = new Float32Array(16);
      const base = best * 256;
      for (let r = 0; r < 16; r++) { let t = 0; for (let c = 0; c <= r; c++) t += Lc[base + r * 16 + c] * z[c]; nv[r] = t; }
    }
    for (let p = 0; p < 16; p++) {
      const y = i + (p >> 2), x = j + (p & 3);
      out[y * 32 + x] += v[p] * sd; cnt[y * 32 + x] += 1;
      if (nv) noise[y * 32 + x] += nv[p] * sd;
    }
  }
  for (let k = 0; k < 1024; k++) out[k] = out[k] / cnt[k] + (sample ? noise[k] / Math.sqrt(cnt[k]) : 0);
  return out;
}

// Layout -> stages [lift, +r2, +r1, +r0] and the three bands, all (32,32,3) interleaved.
export function refine(M, L, arm = "P3", rand = Math.random) {
  const C = liftLayout(M, L);
  const stages = [C], bands = [];
  let cur = C;
  for (const b of [2, 1, 0]) {
    const r = new Float32Array(3072), next = new Float32Array(3072);
    for (let ch = 0; ch < 3; ch++) {
      const plane = new Float32Array(1024);
      for (let k = 0; k < 1024; k++) plane[k] = cur[k * 3 + ch];
      const rb = predictBand(M, plane, b, arm, rand);
      for (let k = 0; k < 1024; k++) { r[k * 3 + ch] = rb[k]; next[k * 3 + ch] = cur[k * 3 + ch] + rb[k]; }
    }
    bands.push(r); stages.push(next); cur = next;
  }
  return { stages, bands, image: cur };
}

// PIL-style bicubic upsampling 8 -> 32 (Keys kernel a = -0.5, pixel-centre mapping, edge clamp)
export function bicubic(L) {
  const k = (t) => { t = Math.abs(t); const a = -0.5;
    return t < 1 ? (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1 : t < 2 ? a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a : 0; };
  const out = new Float32Array(3072);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
    const sy = (y + 0.5) / 4 - 0.5, sx = (x + 0.5) / 4 - 0.5;
    const y0 = Math.floor(sy), x0 = Math.floor(sx);
    for (let c = 0; c < 3; c++) {
      let s = 0, ws = 0;
      for (let m = -1; m <= 2; m++) for (let n = -1; n <= 2; n++) {
        const w = k(sy - (y0 + m)) * k(sx - (x0 + n));
        const yy = Math.min(7, Math.max(0, y0 + m)), xx = Math.min(7, Math.max(0, x0 + n));
        s += w * L[(yy * 8 + xx) * 3 + c]; ws += w;
      }
      out[(y * 32 + x) * 3 + c] = s / ws;
    }
  }
  return out;
}

export function layoutOf(rgb8) {             // uint8 (32,32,3) -> 192 floats
  const L = new Float32Array(192);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) for (let c = 0; c < 3; c++)
    L[(((y >> 2) * 8) + (x >> 2)) * 3 + c] += rgb8[(y * 32 + x) * 3 + c] / 255 / 16;
  return L;
}

export function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
