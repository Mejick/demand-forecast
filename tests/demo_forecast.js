// forecast with docs/core.js in Node, used by test_demo.py
const fs = require("fs");
const path = require("path");
const core = require(path.join(__dirname, "..", "docs", "core.js"));

core.setModel(JSON.parse(fs.readFileSync(path.join(__dirname, "..", "docs", "model.json"), "utf8")));
const lines = fs.readFileSync(process.argv[2], "utf8").split(/\r?\n/).filter((l) => l && !l.startsWith("#"));
const head = lines[0].split(",");
const col = (name) => head.indexOf(name);
const rows = lines.slice(1).map((l) => {
  const v = l.split(",");
  return { date: v[col("date")], sku: v[col("sku")], sold: +v[col("sold")] };  // no stock column
});
const out = {};
for (const s of core.buildSeries(rows)) {
  if (s.launch < 0 || s.n < 56) continue;
  out[s.sku] = Array.from({ length: 28 }, (_, h) => core.treeScore(core.features(s, s.n + h)));
}
process.stdout.write(JSON.stringify(out));
