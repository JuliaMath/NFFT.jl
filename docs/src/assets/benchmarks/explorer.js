// Benchmark explorer of the NFFT.jl documentation. Active on pages with <div id="nfft-explorer">;
// loads Plotly and the measurements (data.js, written by `julia benchmark/bench.jl report`).
(function () {
  const base = document.currentScript.src.replace(/explorer\.js.*$/, "");
  // one colour per library family on every device, as in the figures
  const COLORS = { "NFFT": "#1f4e9c", "FINUFFT": "#d9822b", "ducc": "#3a9e5f",
                   "NonuniformFFTs": "#8e5bb5", "NFFT 0.14": "#9a9a9a", "mlx-nufft": "#c0395a" };
  const colorOf = lib => COLORS[lib === "cuFINUFFT" ? "FINUFFT" : lib.replace(/ (CUDA|Metal)$/, "")] || "#555";
  const MACHINES = { "apple-m1-max": "Apple M1 Max", "amd-epyc-7763": "AMD EPYC 7763 (Zen 3)",
                     "rtx-a6000": "Nvidia RTX A6000" };
  const DASH = { nfft: "solid", rnfft: "dash", nfct: "dot", nfst: "dashdot", nnfft: "solid" };
  const SUITES = {
    pareto:  { x: "err", sweep: ["tol"], label: "time against measured error" },
    cuda:    { x: "err", sweep: ["tol", "threads"], label: "CUDA: time against measured error" },
    metal:   { x: "err", sweep: ["tol", "threads"], label: "Metal: time against measured error" },
    threads: { x: "threads", sweep: ["threads"], label: "strong scaling" },
    sizes:   { x: "size", sweep: ["N"], y: "per_node", label: "problem size" },
    nodes:   { x: "nodes", sweep: ["nodes"], label: "node distributions" },
    batch:   { x: "L", sweep: ["L"], y: "per_transform", label: "batches (OCT, MRI, volumes)" },
    real:    { x: "transform", sweep: ["transform"], label: "real data" },
    nnfft:   { x: "err", sweep: ["tol"], label: "NNFFT (type 3): time against measured error" },
    history: { x: "tol", sweep: ["tol"], label: "NFFT.jl 0.14 and now" },
  };
  const XAXES = {
    err: { label: "measured relative error", log: true, reversed: true, f: (r, op) => op === "t_adj" ? r.err_adj : r.err_trafo },
    tol: { label: "requested tolerance", log: true, reversed: true, f: r => r.tol },
    threads: { label: "threads", log: true, f: r => r.threads },
    size: { label: "|N|", log: true, f: r => r.N.reduce((a, b) => a * b, 1) },
    L: { label: "transforms in the batch", log: true, f: r => r.L },
    nodes: { label: "node distribution", cat: true, f: r => r.nodes },
    transform: { label: "transform", cat: true, f: r => r.transform },
  };
  const YAXES = {
    t_trafo: "direct transform / ms", t_adj: "adjoint transform / ms", t_plan: "planning / ms",
    per_node: "direct + adjoint per node / ns", per_transform: "direct + adjoint per transform / ms",
  };
  const FILTERS = ["machine", "case", "D", "N", "T", "threads", "tol", "nodes", "L", "transform", "lib"];
  const LABELS = { machine: "machine", T: "precision", N: "size" };

  function load(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = fail; document.head.appendChild(s);
    });
  }

  function yval(r, y) {
    if (y === "per_node") return 1e9 * (r.t_trafo + r.t_adj) / r.J / r.L;
    if (y === "per_transform") return 1e3 * (r.t_trafo + r.t_adj) / r.L;
    return 1e3 * r[y];
  }
  const caseOf = r => r.suite.includes(":") ? r.suite.split(":")[1] : "";
  const fieldOf = (r, f) => f === "case" ? caseOf(r) : f === "N" ? r.N.join("×") : r[f];
  const fmt = v => typeof v === "number" ? (Math.abs(v) < 1e-2 || Math.abs(v) >= 1e5 ? v.toExponential(1) : +v.toPrecision(3)) : v;

  function el(tag, attrs, ...children) {
    const e = document.createElement(tag);
    Object.assign(e, attrs || {});
    children.forEach(c => e.append(c));
    return e;
  }

  function init(root) {
    const rows = window.NFFT_BENCH.rows;
    const state = { suite: "pareto", y: "t_trafo", filters: { machine: window.NFFT_BENCH.reference } };
    const controls = el("div", { className: "nfft-explorer-controls" });
    const plot = el("div", { className: "nfft-explorer-plot" });
    const note = el("p", { className: "nfft-explorer-note" });
    root.append(controls, plot, note);

    function select(label, options, value, onchange) {
      const s = el("select");
      options.forEach(([v, t]) => s.append(el("option", { value: v, textContent: t, selected: String(v) === String(value) })));
      s.onchange = () => onchange(s.value);
      return el("label", {}, label + " ", s);
    }

    function render() {
      const suiteRows = rows.filter(r => r.suite.split(":")[0] === state.suite);
      const xa = XAXES[SUITES[state.suite].x];
      controls.replaceChildren();
      controls.append(select("Suite", Object.entries(SUITES).filter(([k]) => rows.some(r => r.suite.split(":")[0] === k))
                                        .map(([k, v]) => [k, v.label]), state.suite,
                             v => { state.suite = v; state.filters = { machine: state.filters.machine };
                                    state.y = SUITES[v].y || "t_trafo"; render(); }));
      controls.append(select("y", Object.entries(YAXES), state.y, v => { state.y = v; render(); }));
      // a filter for every field that varies and is not the x axis or the series
      let sel = suiteRows;
      for (const f of FILTERS) {
        if (SUITES[state.suite].sweep.includes(f) || f === "lib") continue;
        const vals = [...new Set(sel.map(r => fieldOf(r, f)))].sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
        if (vals.length < 2) continue;
        if (state.filters[f] === undefined || !vals.map(String).includes(String(state.filters[f]))) {
          const pref = { machine: window.NFFT_BENCH.reference, T: "Float64", nodes: "uniform", tol: 1e-8,
                         threads: Math.max(...vals), L: 1 }[f];
          state.filters[f] = vals.includes(pref) ? pref : vals[0];
        }
        controls.append(select(LABELS[f] || f, vals.map(v => [v, f === "machine" ? MACHINES[v] || v : fmt(v)]), state.filters[f],
                               v => { state.filters[f] = isNaN(+v) || v === "" ? v : +v; render(); }));
        sel = sel.filter(r => String(fieldOf(r, f)) === String(state.filters[f]));
      }
      const op = state.y === "t_adj" ? "t_adj" : "t_trafo";
      const series = {};
      for (const r of sel) {
        const name = state.suite === "real" || state.suite === "nnfft" ? r.lib : r.lib + (r.transform !== "nfft" ? " " + r.transform.toUpperCase() : "");
        (series[name] = series[name] || []).push(r);
      }
      const traces = Object.entries(series).map(([name, rs]) => {
        if (!xa.cat) rs.sort((a, b) => xa.f(a, op) - xa.f(b, op));
        return {
          name, type: xa.cat ? "bar" : "scatter", mode: "lines+markers",
          x: rs.map(r => xa.f(r, op)), y: rs.map(r => yval(r, state.y)),
          marker: { color: colorOf(rs[0].lib) },
          line: { color: colorOf(rs[0].lib), dash: DASH[rs[0].transform] || "solid" },
          text: rs.map(r => [`${r.lib} ${r.transform.toUpperCase()}, ${r.D}D, N = ${r.N.join("×")}, J = ${r.J}` + (r.L > 1 ? `, L = ${r.L}` : ""),
                             `${r.T}, ${r.threads} threads, ${r.nodes} nodes, tol ${fmt(r.tol)}`,
                             `error ${fmt(r.err_trafo)} / ${fmt(r.err_adj)}`,
                             `plan ${fmt(1e3 * r.t_plan)} ms, direct ${fmt(1e3 * r.t_trafo)} ms, adjoint ${fmt(1e3 * r.t_adj)} ms`]
                            .concat(r.W ? [`W = ${r.W}, σ = ${fmt(r.sigma)}`] : []).join("<br>")),
          hovertemplate: "%{text}<extra></extra>",
        };
      });
      Plotly.react(plot, traces, {
        paper_bgcolor: "white", plot_bgcolor: "white", height: 460, margin: { t: 20, r: 10, b: 50, l: 70 },
        font: { family: "Lato, Helvetica, Arial, sans-serif", size: 13 },
        xaxis: { title: xa.label, type: xa.cat ? "category" : xa.log ? "log" : "linear",
                 autorange: xa.reversed ? "reversed" : true, gridcolor: "#eee", exponentformat: "power" },
        yaxis: { title: YAXES[state.y], type: "log", gridcolor: "#eee", exponentformat: "power" },
        legend: { orientation: "h", y: -0.2 }, barmode: "group",
      }, { displaylogo: false, responsive: true });
      const machine = sel.length ? sel[0].machine : state.filters.machine;
      const meta = window.NFFT_BENCH.meta[machine + "/" + state.suite] || {};
      note.textContent = `${sel.length} measurements · ${meta.machine || ""}, Julia ${meta.julia || ""}, ` +
                         `NFFT.jl ${meta.commit || ""}, ${(meta.date || "").slice(0, 10)}`;
    }
    render();
  }

  // Machine selector for the CPU figures: <div class="nfft-machine-switch" data-figures="accuracy density">
  // switches every figure assets/benchmarks/<name>.svg on the page to <name>-<machine>.svg and back.
  function machineSwitch(div) {
    const names = div.dataset.figures.split(" ");
    const options = JSON.parse(div.dataset.machines);
    const imgs = [...document.querySelectorAll("img")].filter(img => names.some(n => img.src.match(new RegExp("/" + n + "(-[a-z0-9-]+)?\\.svg$"))));
    const s = el("select");
    Object.entries(options).forEach(([v, t], i) => s.append(el("option", { value: v, textContent: t, selected: i === 0 })));
    s.onchange = () => imgs.forEach(img => {
      img.src = img.src.replace(/\/([a-z]+)(-[a-z0-9-]+)?\.svg$/, (m, n) => "/" + n + (s.value === "" ? "" : "-" + s.value) + ".svg");
    });
    div.append(el("label", {}, "Figures of this section: ", s));
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".nfft-machine-switch").forEach(machineSwitch);
    const root = document.getElementById("nfft-explorer");
    if (!root) return;
    // Documenter loads RequireJS, with which Plotly registers as an AMD module instead of a global
    const plotly = "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@2.35.2/plotly-basic.min.js";
    const loadPlotly = typeof window.require === "function" && window.define && window.define.amd
      ? new Promise((ok, fail) => window.require([plotly], P => { window.Plotly = P; ok(); }, fail))
      : load(plotly);
    Promise.all([loadPlotly, load(base + "data.js")]).then(() => init(root))
      .catch(e => { console.error("NFFT explorer:", e); root.textContent = "The explorer needs JavaScript and the Plotly library."; });
  });
})();
