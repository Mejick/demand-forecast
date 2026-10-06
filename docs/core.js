// forecast math for the page and the test, same features as src/features.py
"use strict";

var H = 28, SHIFT = 28, K_WEEK = 1.5, DISP_WINDOW = 112;
var MODEL = null;  // {features, trees}

// model
function treeScore(x) {
  let s = 0;
  for (const tr of MODEL.trees) {
    let n = tr.root;
    while (n >= 0) {
      const v = x[tr.f[n]];
      const left = Number.isNaN(v) ? tr.d[n] === 1 : v <= tr.t[n];
      n = left ? tr.l[n] : tr.r[n];
    }
    s += tr.v[~n];
  }
  return Math.exp(s);
}

// features for future day x
function features(s, x) {
  const pos = x;
  const nan = NaN;
  const win = (arr, w) => {  // sum over w days ending 28 days earlier
    if (pos < SHIFT) return nan;
    const hi = Math.min(SHIFT + w - 1, pos);
    let t = 0;
    for (let i = x - hi; i <= x - SHIFT; i++) t += arr[i];
    return t;
  };
  const f = {};
  for (const k of [28, 35]) f["lag_" + k] = pos >= k && s.launched[x - k] ? s.y[x - k] : nan;
  for (const w of [7, 28, 56]) {
    const cnt = win(s.c, w);
    f["mean_" + w] = cnt > 0 ? win(s.v, w) / cnt : nan;
  }
  const c28 = win(s.c, 28), m28 = f.mean_28;
  f.std_28 = c28 > 1 ? Math.sqrt(Math.max(win(s.v2, 28) / c28 - m28 * m28, 0)) : nan;
  for (const w of [28, 56]) {
    const cnt = win(s.c, w);
    f["sale_share_" + w] = cnt > 0 ? win(s.ps, w) / cnt : nan;
  }
  const ns = win(s.ps, 56);
  f.size_56 = ns > 0 ? win(s.v, 56) / ns : nan;
  const dow = [28, 35, 42, 49].filter((k) => pos >= k && s.instock[x - k]).map((k) => s.y[x - k]);
  f.dow_mean_4 = dow.length ? dow.reduce((a, b) => a + b, 0) / dow.length : nan;
  const d = s.date(x);
  f.wday = (d.getUTCDay() + 6) % 7;
  f.weekend = f.wday >= 5 ? 1 : 0;
  f.mday = d.getUTCDate();
  f.month = d.getUTCMonth() + 1;
  const age = s.launch >= 0 ? x - s.launch : -1;
  f.age_days = Math.min(age, 90);
  f.is_new = age >= 0 && age < 28 ? 1 : 0;
  return MODEL.features.map((k) => f[k]);
}

// negative binomial
function nbParams(m, v) {
  m = Math.max(m, 1e-9);
  if (v <= m * (1 + 1e-6)) return { r: 1e6, p: 1e6 / (1e6 + m) };  // almost Poisson
  const r = (m * m) / (v - m);
  return { r, p: r / (r + m) };
}
function nbCdfTo(m, v, k) {  // P(X <= k)
  const { r, p } = nbParams(m, v);
  let pmf = Math.exp(r * Math.log(p)), cdf = pmf;
  for (let i = 0; i < k; i++) { pmf *= ((i + r) / (i + 1)) * (1 - p); cdf += pmf; }
  return Math.min(cdf, 1);
}
function nbQuantile(m, v, q) {
  const { r, p } = nbParams(m, v);
  let pmf = Math.exp(r * Math.log(p)), cdf = pmf, k = 0;
  while (cdf < q && k < 1e6) { pmf *= ((k + r) / (k + 1)) * (1 - p); cdf += pmf; k++; }
  return k;
}
function sumDist(mu, r, a, b) {
  let m = 0, vd = 0;
  for (let i = a; i < Math.min(b, mu.length); i++) { m += mu[i]; vd += mu[i] + (mu[i] * mu[i]) / r; }
  return { m, v: m + K_WEEK * (vd - m) };
}

// daily series per item
function buildSeries(rows) {
  const by = new Map();
  let maxT = -Infinity;
  for (const r of rows) {
    const t = Date.parse(r.date + "T00:00:00Z");
    if (!isFinite(t)) continue;
    maxT = Math.max(maxT, t);
    if (!by.has(r.sku)) by.set(r.sku, []);
    by.get(r.sku).push({ t, ...r });
  }
  const DAY = 86400000, out = [];
  for (const [sku, list] of by) {
    list.sort((a, b) => a.t - b.t);
    const t0 = list[0].t, n = Math.round((maxT - t0) / DAY) + 1;
    const y = new Float64Array(n), stock = new Float64Array(n).fill(NaN), price = new Float64Array(n).fill(NaN);
    for (const r of list) {
      const i = Math.round((r.t - t0) / DAY);
      y[i] += r.sold || 0;
      if (r.stock != null && isFinite(r.stock)) stock[i] = r.stock;
      if (r.price != null && isFinite(r.price)) price[i] = r.price;
    }
    const hasStock = list.some((r) => r.stock != null && isFinite(r.stock));
    const launch = y.findIndex((v) => v > 0);
    const launched = Array.from(y, (_, i) => launch >= 0 && i >= launch);
    // zero with empty shelf is not demand
    const instock = launched.map((L, i) => L && !(hasStock && y[i] === 0 && (i > 0 ? stock[i - 1] : stock[i]) <= 0));
    const v = Array.from(y, (val, i) => (instock[i] ? val : 0));
    const s = {
      sku, category: list[0].category || "", n, y, stock, hasStock, launch, launched, instock,
      v, v2: v.map((a) => a * a), c: instock.map((b) => (b ? 1 : 0)),
      ps: instock.map((b, i) => (b && y[i] > 0 ? 1 : 0)),
      date: (i) => new Date(t0 + i * DAY),
      lastPrice: [...price].reverse().find((p) => isFinite(p)),
      currentStock: hasStock ? [...stock].reverse().find((p) => isFinite(p)) : null,
    };
    out.push(s);
  }
  return out;
}

if (typeof module !== "undefined") module.exports = { setModel: (m) => { MODEL = m; }, treeScore, features, buildSeries, nbQuantile, nbCdfTo, sumDist };
