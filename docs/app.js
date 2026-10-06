// demo page, everything stays in the browser
"use strict";

let ITEMS = [], CHART = null;

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 0) => (x == null || !isFinite(x)) ? "—" : x.toLocaleString("ru-RU", { maximumFractionDigits: d, minimumFractionDigits: d });

// forecast is cached, risk depends on stock and settings
const STOCK_EDIT = new Map();  // stock typed by the user
let RES = [];

function forecast(s) {
  if (s._fc !== undefined) return s._fc;
  if (s.launch < 0 || s.n < 56) return (s._fc = null);
  const mu = [];
  for (let h = 0; h < H; h++) mu.push(treeScore(features(s, s.n + h)));
  // spread from the last 112 on-sale days
  const hv = [];
  for (let i = Math.max(0, s.n - DISP_WINDOW); i < s.n; i++) if (s.instock[i]) hv.push(s.y[i]);
  let r = 1e3;
  if (hv.length > 1) {
    const m = hv.reduce((a, b) => a + b, 0) / hv.length;
    const v = hv.reduce((a, b) => a + (b - m) ** 2, 0) / hv.length;
    r = v - m > 1e-6 ? (m * m) / (v - m) : 1e3;
  }
  r = Math.min(Math.max(r, 0.05), 1e3);
  const daily = mu.map((m) => { const v = m + (m * m) / r; return { m, lo: nbQuantile(m, v, 0.1), hi: nbQuantile(m, v, 0.9) }; });
  const w7 = sumDist(mu, r, 0, 7);
  return (s._fc = { mu, r, daily, week: w7.m, weekLo: nbQuantile(w7.m, w7.v, 0.1), weekHi: nbQuantile(w7.m, w7.v, 0.9) });
}

function analyse(s, lead, q) {
  const fc = forecast(s);
  if (!fc) return null;
  const stock = STOCK_EDIT.has(s.sku) ? STOCK_EDIT.get(s.sku) : s.currentStock;
  const res = { s, ...fc, stock, edited: STOCK_EDIT.has(s.sku), status: null, pL: null, coverDays: null,
    orderQty: null, orderBy: null, money: s.lastPrice ? fc.week * s.lastPrice : fc.week };
  if (stock == null || !isFinite(stock)) return res;
  const { mu, r } = fc, st = stock;
  const dL = sumDist(mu, r, 0, lead), dW = sumDist(mu, r, 0, lead + 7);
  res.pL = st <= 0 ? 1 : 1 - nbCdfTo(dL.m, dL.v, Math.floor(st));
  res.pW = st <= 0 ? 1 : 1 - nbCdfTo(dW.m, dW.v, Math.floor(st));
  res.status = st <= 0 ? "black" : res.pL >= 0.5 ? "red" : res.pW >= 0.1 ? "yellow" : "green";
  let cum = 0, cover = null;
  for (let h = 0; h < H; h++) { cum += mu[h]; if (cum >= st) { cover = h; break; } }
  res.coverDays = cover;
  res.orderQty = Math.max(0, Math.ceil(nbQuantile(dW.m, dW.v, q) - st));
  if (res.status !== "green") {
    const by = cover == null ? H - lead : cover - lead;
    res.orderBy = by <= 0 ? "сегодня" : s.date(s.n - 1 + by).toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
  }
  return res;
}

// table and chart
const LEVEL = {
  black: { icon: "⚫", name: "Полка пустая", rank: 0 },
  red: { icon: "🔴", name: "Заказать сегодня", rank: 1 },
  yellow: { icon: "🟡", name: "Заказать в ближайшие дни", rank: 2 },
  green: { icon: "🟢", name: "Запаса хватает", rank: 3 },
};

function cells(r) {
  return [
    r.status ? `<span class="lvl lvl-${r.status}">${LEVEL[r.status].icon} ${LEVEL[r.status].name}</span>` : `<span class="muted">впишите остаток →</span>`,
    `<b>${r.s.sku}</b><small>${r.s.category}</small>`,
    `<input class="stockin${r.edited ? " edited" : ""}" type="number" min="0" step="1" value="${r.stock == null ? "" : r.stock}" placeholder="—" aria-label="Остаток ${r.s.sku}">`,
    `${fmt(r.week, 1)}<small>${fmt(r.weekLo)}–${fmt(r.weekHi)}</small>`,
    r.coverDays == null ? (r.status ? "> 28" : "—") : fmt(r.coverDays),
    r.pL == null ? "—" : `${fmt(r.pL * 100)}%`,
    r.orderQty == null ? "—" : r.orderQty ? fmt(r.orderQty) : "не нужно",
    r.orderBy ?? "—",
  ];
}
const CELL_CLASS = ["", "", "num", "num", "num", "num", "num", ""];

function renderSummary() {
  const n = RES.filter((r) => r.status).length;
  const chips = n ? Object.entries(LEVEL).map(([k, L]) => `<div class="chip lvl-${k}"><b>${RES.filter((r) => r.status === k).length}</b>${L.icon} ${L.name}</div>`).join("") : "";
  const hint = n < RES.length ? `<div class="chip">${n ? `Без остатка: ${RES.length - n}. ` : "В файле нет остатков. "}Впишите остаток в колонку «Остаток» — статус и заказ посчитаются сразу.</div>` : "";
  $("summary").innerHTML = chips + hint;
}

function updateRow(tr) {
  const i = +tr.dataset.i;
  const r = analyse(RES[i].s, +$("lead").value, +$("service").value);
  RES[i] = r;
  const tds = tr.children;
  cells(r).forEach((h, k) => { if (k !== 2) tds[k].innerHTML = h; });  // keep focus in the input
  tds[2].querySelector("input").classList.toggle("edited", r.edited);
  renderSummary();
  if ($("chartTitle").dataset.sku === r.s.sku) drawItem(r);
}

function render() {
  const lead = +$("lead").value, q = +$("service").value;
  RES = ITEMS.map((s) => analyse(s, lead, q)).filter(Boolean);
  RES.sort((a, b) => (LEVEL[a.status]?.rank ?? 9) - (LEVEL[b.status]?.rank ?? 9) || b.money - a.money);
  $("thRisk").innerHTML = `Риск пустой полки до прихода заказа<small>заказ сегодня придёт через ${lead} дн.</small>`;
  $("thOrder").innerHTML = `Заказать, шт.<small>на ${lead} + 7 дн., сервис ${Math.round(q * 100)}%</small>`;
  renderSummary();
  const last = ITEMS[0] ? ITEMS[0].date(ITEMS[0].n - 1).toLocaleDateString("ru-RU") : "";
  $("meta").textContent = `${RES.length} товаров · данные по ${last} · прогноз на 28 дней вперёд`;
  $("rows").innerHTML = RES.map((r, i) => `<tr data-i="${i}">${cells(r).map((c, k) => `<td class="${CELL_CLASS[k]}">${c}</td>`).join("")}</tr>`).join("");
  $("results").hidden = false;
  for (const tr of $("rows").querySelectorAll("tr")) {
    tr.addEventListener("click", (e) => {
      if (e.target.closest("input")) return;
      drawItem(RES[+tr.dataset.i]);
      $("chartTitle").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    const inp = tr.querySelector("input");
    inp.addEventListener("input", () => {
      const v = inp.value.trim(), sku = RES[+tr.dataset.i].s.sku;
      if (v === "") STOCK_EDIT.delete(sku); else STOCK_EDIT.set(sku, Math.max(0, Math.round(+v)));
      updateRow(tr);
    });
  }
  if (RES.length) drawItem(RES[0]);
}

function drawItem(r) {
  const s = r.s, from = Math.max(0, s.n - 120);
  const labels = [], hist = [], fc = [], lo = [], hi = [];
  for (let i = from; i < s.n; i++) { labels.push(s.date(i).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" })); hist.push(s.y[i]); fc.push(null); lo.push(null); hi.push(null); }
  r.daily.forEach((d, h) => { labels.push(s.date(s.n + h).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" })); hist.push(null); fc.push(d.m); lo.push(d.lo); hi.push(d.hi); });
  $("chartTitle").textContent = `${s.sku} · ${s.category} — продажи за 120 дней и прогноз на 28`;
  $("chartTitle").dataset.sku = s.sku;
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ink-2").trim(), accent = css.getPropertyValue("--accent").trim(), line = css.getPropertyValue("--line").trim();
  if (CHART) CHART.destroy();
  CHART = new Chart($("chart"), {
    type: "line",
    data: { labels, datasets: [
      { label: "Продажи", data: hist, borderColor: ink, borderWidth: 1.2, pointRadius: 0 },
      { label: "Диапазон 80%", data: hi, borderColor: "transparent", backgroundColor: accent + "33", fill: "+1", pointRadius: 0 },
      { label: "", data: lo, borderColor: "transparent", pointRadius: 0, fill: false },
      { label: "Прогноз", data: fc, borderColor: accent, borderWidth: 2, pointRadius: 0 },
    ] },
    options: { animation: false, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: ink, filter: (i) => i.text } } },
      scales: { x: { ticks: { color: ink, maxTicksLimit: 10 }, grid: { color: line } }, y: { beginAtZero: true, ticks: { color: ink }, grid: { color: line } } } },
  });
}

// file input
const ALIASES = {
  date: ["date", "дата", "day", "день"], sku: ["sku", "товар", "item", "артикул", "id"],
  sold: ["sold", "продано", "qty", "quantity", "sales", "количество"], stock: ["stock", "остаток", "on_hand"],
  price: ["price", "цена"], category: ["category", "категория", "группа"],
};
function normalise(rows) {
  if (!rows.length) return [];
  const keys = Object.keys(rows[0]), map = {};
  for (const [k, al] of Object.entries(ALIASES)) map[k] = keys.find((c) => al.includes(c.trim().toLowerCase()));
  if (!map.date || !map.sku || !map.sold) throw new Error("Нужны колонки: дата, товар и сколько продано (date, sku, sold).");
  const num = (v) => (v === "" || v == null ? null : +String(v).replace(",", "."));
  return rows.map((r) => ({ date: String(r[map.date]).slice(0, 10), sku: String(r[map.sku]), sold: num(r[map.sold]) || 0,
    stock: map.stock ? num(r[map.stock]) : null, price: map.price ? num(r[map.price]) : null, category: map.category ? r[map.category] : "" }));
}
function load(text, label) {
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true, comments: "#" });
  try {
    ITEMS = buildSeries(normalise(parsed.data));
    STOCK_EDIT.clear();
    $("source").textContent = label;
    $("error").hidden = true;
    render();
  } catch (e) { $("error").textContent = e.message; $("error").hidden = false; }
}

async function init() {
  MODEL = await (await fetch("model.json")).json();
  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) f.text().then((t) => load(t, `Ваш файл: ${f.name}`));
  });
  $("sample").addEventListener("click", async () => load(await (await fetch("sample_synthetic_store.csv")).text(),
    "Пример: синтетические данные — сгенерированный магазин, не реальные продажи"));
  for (const id of ["lead", "service"]) $(id).addEventListener("change", () => ITEMS.length && render());
  $("ready").hidden = false;
}
init().catch((e) => { $("error").textContent = "Не удалось загрузить модель: " + e.message; $("error").hidden = false; });
