// Checks web/model.js against the Python model (weights are float16 in the browser, so small differences are expected).
import { readFileSync } from "node:fs";
import { loadModel, newSequence, stepLogits, tokensToLayout, liftLayout, refine, bicubic } from "../web/model.js";

const root = new URL("..", import.meta.url).pathname;
const meta = JSON.parse(readFileSync(root + "web/model.json"));
const buf = readFileSync(root + "web/model.bin");
const M = loadModel(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const ref = JSON.parse(readFileSync(root + "tests/parity_ref.json"));

const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const meanAbs = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;
let fails = 0;
const check = (name, ok, info) => { console.log(`${ok ? "ok  " : "FAIL"} ${name} ${info}`); if (!ok) fails++; };

const seq = newSequence(M);
const got = {};
for (let i = 0; i < 64; i++) {
  const lg = stepLogits(M, seq, ref.cls);
  if (ref.logit_pos.includes(i)) got[i] = lg;
  seq.toks.push(ref.toks[i]); seq.pos++;
}
ref.logit_pos.forEach((p, k) => {
  const d = maxAbs(got[p], ref.logits[k]);
  const am = (a) => a.indexOf(Math.max(...a));
  check(`logits at position ${p}`, d < 0.05 && am(Array.from(got[p])) === am(ref.logits[k]), `max |diff| ${d.toExponential(2)}`);
});
const L = tokensToLayout(M, ref.toks);
check("layout from tokens", maxAbs(L, ref.layout) < 2e-3, maxAbs(L, ref.layout).toExponential(2));
const C = liftLayout(M, Float32Array.from(ref.layout));
check("lift", maxAbs(C, ref.lift) < 5e-3, maxAbs(C, ref.lift).toExponential(2));
for (const [arm, key] of [["P3", "p3"], ["P0", "p0"]]) {
  const im = refine(M, Float32Array.from(ref.layout), arm).image;
  const m = meanAbs(im, ref[key]);
  check(`refine ${arm}`, m < 2e-3, `mean |diff| ${m.toExponential(2)}, max ${maxAbs(im, ref[key]).toExponential(2)}`);
}
const bc = bicubic(Float32Array.from(ref.layout));
const bm = meanAbs(bc, ref.bicubic);
check("bicubic (PIL rounds to uint8)", bm < 5e-3, `mean |diff| ${bm.toExponential(2)}`);
process.exit(fails ? 1 : 0);
