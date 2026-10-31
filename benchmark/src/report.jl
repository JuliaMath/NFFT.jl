# Figures and data of the documentation from saved results (results/<machine>/*.csv):
#
#   julia benchmark/bench.jl report [machine]
#
# writes docs/src/assets/benchmarks/*.svg and data.js (all measurements, for the explorer).

include(joinpath(@__DIR__, "Benchmarks.jl"))
using .Benchmarks: read_csv, ok, ROOT, REPO
using CairoMakie

const MACHINE = ARGS[1]
const SRC = joinpath(ROOT, "results", MACHINE)
const OUT = joinpath(REPO, "docs", "src", "assets", "benchmarks")
mkpath(OUT)

const COLORS = Dict("NFFT" => "#1f4e9c", "FINUFFT" => "#d9822b", "ducc" => "#3a9e5f",
                    "NonuniformFFTs" => "#8e5bb5", "NFFT 0.14" => "#9a9a9a")
const LIBORDER = ["NFFT", "FINUFFT", "ducc", "NonuniformFFTs", "NFFT 0.14"]
color(lib) = get(COLORS, lib, "#555555")

function load(suite)
  file = joinpath(SRC, "$suite.csv")
  isfile(file) || return Dict{String,String}(), Dict{String,Any}[]
  meta, rows = read_csv(file)
  meta, filter(ok, rows)
end

dimlabel(r) = r["transform"] == "nnfft" ? "$(r["D"])D, $(r["J"]) sources and targets" :
              r["D"] == 1 ? "1D, N = $(r["N"][1])" : "$(r["D"])D, N = $(join(r["N"], "×"))"
libs_in(rows) = filter(l -> any(r -> r["lib"] == l, rows), LIBORDER)
ms(t) = 1e3t
set_theme!(Theme(fontsize=13, Axis=(xgridstyle=:dash, ygridstyle=:dash, xgridcolor=(:black, 0.08),
                                    ygridcolor=(:black, 0.08))))

# Log axes: powers of ten as 10⁻⁴ (at most 7), values as 0.5, 1, 2, 5 (1–2–5 or sparser)
const SUP = Dict(zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
pow10(e) = "10" * map(c -> SUP[c], string(e))
function exp_ticks(lo, hi)
  es = ceil(Int, log10(lo) - 1e-9):floor(Int, log10(hi) + 1e-9)
  [10.0^e for e in es[1:cld(length(es), 7):end]]
end
exp_format(vs) = [pow10(round(Int, log10(v))) for v in vs]
function val_ticks(lo, hi)
  for ms in ([1, 2, 3, 5, 7], [1, 2, 5], [1, 3], [1])
    ts = [m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in ms]
    ts = filter(t -> lo <= t <= hi, ts)
    length(ts) <= 6 && return length(ts) >= 2 || ms == [1] ? ts : val_ticks_fallback(lo, hi)
  end
  [10.0^e for e in ceil(Int, log10(lo)):floor(Int, log10(hi))]
end
val_ticks_fallback(lo, hi) = [lo * (hi / lo)^(i / 3) for i in 0:3] .|> v -> round(v, sigdigits=2)
val_format(vs) = [v >= 1 && v == round(v) ? string(round(Int, v)) : string(round(v, sigdigits=2)) for v in vs]
const LOGX_EXP = (xscale=log10, xticks=exp_ticks, xtickformat=exp_format)
const LOGY_VAL = (yscale=log10, yticks=val_ticks, ytickformat=val_format)

function legend_below!(fig, row, libs; extra=[])
  els = [[LineElement(color=color(l), linewidth=2), MarkerElement(color=color(l), marker=:circle)] for l in libs]
  Legend(fig[row, :], [els; first.(extra)], [libs; last.(extra)]; orientation=:horizontal, framevisible=false)
end

save_fig(name, fig) = (save(joinpath(OUT, name * ".svg"), fig); println("  ", name, ".svg"))

# Speed of the other libraries relative to NFFT.jl at equal measured error: time of a library over
# the time of NFFT.jl at the error that library delivers, interpolated log-log between NFFT.jl's
# measurements of the same case (none outside their range). Rows direct and adjoint, columns D,
# the y axes linked; > 1: NFFT.jl faster.
const OPS = (("direct", "err_trafo", "t_trafo"), ("adjoint", "err_adj", "t_adj"))

function time_at(own, e, err, t)
  pts = sort([(r[err], r[t]) for r in own if r[err] > 0]; by=first)
  for k in 1:length(pts) - 1
    (e1, t1), (e2, t2) = pts[k], pts[k + 1]
    e1 <= e <= e2 || continue
    a = e1 == e2 ? 0.0 : log(e / e1) / log(e2 / e1)
    return exp((1 - a) * log(t1) + a * log(t2))
  end
  nothing
end

# case(r): rows with equal case(r) are the same problem; x(r, err): position on the x axis; series(r):
# line style of the series a row belongs to (one line per library and series)
function ratio_figure(name, rows; case, x, xlabel, xaxis=LOGX_EXP, xreversed=false, series=r -> :solid,
                      title=dimlabel, extra=[])
  isempty(rows) && return
  others = filter(!=("NFFT"), libs_in(rows))
  fig = Figure(size=(900, 560)); axes = Axis[]
  for (j, D) in enumerate(sort(unique(r["D"] for r in rows)))
    rd = filter(r -> r["D"] == D, rows)
    for (i, (op, err, t)) in enumerate(OPS)
      ax = Axis(fig[i, j]; xaxis..., yscale=log10, yticks=val_ticks, ytickformat=val_format, xreversed,
                title=i == 1 ? title(rd[1]) : "", xlabel=i == 2 ? xlabel : "",
                ylabel=j == 1 ? "$op: time / time of NFFT.jl" : "")
      push!(axes, ax)
      hlines!(ax, [1]; color=(:black, 0.4), linewidth=1)
      for lib in others, s in unique(series(r) for r in rd)
        pts = Tuple{Float64,Float64}[]
        for r in rd
          r["lib"] == lib && series(r) == s || continue
          tn = time_at(filter(o -> o["lib"] == "NFFT" && case(o) == case(r), rd), r[err], err, t)
          tn === nothing || push!(pts, (x(r, err), r[t] / tn))
        end
        isempty(pts) && continue
        sort!(pts)
        scatterlines!(ax, first.(pts), last.(pts); color=color(lib), linestyle=s, markersize=6)
      end
    end
  end
  linkyaxes!(axes...)
  legend_below!(fig, 3, others; extra)
  save_fig(name, fig)
end

const PRECISION_STYLES = [LineElement(color=:black, linestyle=:solid) => "Float64",
                          LineElement(color=:black, linestyle=:dash) => "Float32"]
precision_style(r) = r["T"] == "Float32" ? :dash : :solid

# speed against accuracy (tolerance sweep), both precisions
accuracy(; suite="pareto", name="accuracy") =
  ratio_figure(name, load(suite)[2]; case=r -> (r["T"],), x=(r, err) -> r[err],
               xlabel="relative error", xreversed=true, series=precision_style, extra=PRECISION_STYLES)

# speed against the node density J/|N|
density() =
  ratio_figure("density", load("density")[2]; case=r -> (r["N"], r["J"]), x=(r, _) -> r["J"] / prod(r["N"]),
               xlabel="nodes per mode J/|N|", title=r -> "$(r["D"])D, tol 1e-8")

# speed against the problem size, J = |N|
sizes() =
  ratio_figure("sizes", load("sizes")[2]; case=r -> (r["N"],), x=(r, _) -> prod(r["N"]), xlabel="|N| = J",
               title=r -> "$(r["D"])D, tol 1e-8")

# strong scaling
function threads()
  _, rows = load("threads")
  isempty(rows) && return
  tol = 1e-8
  rows = filter(r -> r["tol"] == tol, rows)
  fig = Figure(size=(900, 330))
  for (j, D) in enumerate(sort(unique(r["D"] for r in rows)))
    rd = filter(r -> r["D"] == D, rows)
    ps = sort(unique(r["threads"] for r in rd))
    ax = Axis(fig[1, j]; xscale=log2, yscale=log2, xticks=ps, yticks=ps, title=dimlabel(rd[1]),
              xlabel="threads", ylabel=j == 1 ? "speedup" : "")
    lines!(ax, ps, ps; color=(:black, 0.25), linestyle=:dot)
    for lib in libs_in(rd), (t, ls) in (("t_trafo", :solid), ("t_adj", :dash))
      rl = sort(filter(r -> r["lib"] == lib, rd); by=r -> r["threads"])
      t1 = only(r[t] for r in rl if r["threads"] == 1)
      scatterlines!(ax, [r["threads"] for r in rl], [t1 / r[t] for r in rl]; color=color(lib), linestyle=ls, markersize=6)
    end
  end
  legend_below!(fig, 2, libs_in(rows); extra=[LineElement(color=:black, linestyle=:solid) => "direct",
                                               LineElement(color=:black, linestyle=:dash) => "adjoint"])
  save_fig("threads", fig)
end

function batch()
  _, rows = load("batch")
  isempty(rows) && return
  rows = filter(r -> r["tol"] == 1e-8, rows)
  cases = unique(r["suite"] for r in rows)
  names = Dict("batch:oct" => "OCT: 1D, N = 2048, J = 2048", "batch:mri" => "multi-coil MRI: 256², J = 65536",
               "batch:volume" => "volumes: 64³, J = 2¹⁸")
  fig = Figure(size=(900, 330))
  for (j, c) in enumerate(cases)
    rc = filter(r -> r["suite"] == c, rows)
    ax = Axis(fig[1, j]; xscale=log2, LOGY_VAL..., title=get(names, c, c), xlabel="transforms L",
              ylabel=j == 1 ? "time per transform / ms" : "", xticks=sort(unique(r["L"] for r in rc)))
    for lib in libs_in(rc), (t, ls) in (("t_trafo", :solid), ("t_adj", :dash))
      rl = sort(filter(r -> r["lib"] == lib, rc); by=r -> r["L"])
      scatterlines!(ax, [r["L"] for r in rl], [ms(r[t]) / r["L"] for r in rl]; color=color(lib), linestyle=ls, markersize=6)
    end
  end
  legend_below!(fig, 2, libs_in(rows); extra=[LineElement(color=:black, linestyle=:solid) => "direct",
                                               LineElement(color=:black, linestyle=:dash) => "adjoint"])
  save_fig("batch", fig)
end

function real()
  _, rows = load("real")
  isempty(rows) && return
  rows = filter(r -> r["tol"] == 1e-8 && r["suite"] == "real", rows)
  label(r) = r["lib"] == "NFFT" ? (r["transform"] == "nfft" ? "NFFT" : uppercase(r["transform"])) : r["lib"] * " (complex)"
  groups = ["NFFT", "RNFFT", "NFCT", "NFST", "FINUFFT (complex)", "ducc (complex)"]
  gcolor = Dict("NFFT" => COLORS["NFFT"], "RNFFT" => "#4a7fd4", "NFCT" => "#7fa6e3", "NFST" => "#a9c3ec",
                "FINUFFT (complex)" => COLORS["FINUFFT"], "ducc (complex)" => COLORS["ducc"])
  fig = Figure(size=(900, 330))
  for (j, D) in enumerate(sort(unique(r["D"] for r in rows)))
    rd = filter(r -> r["D"] == D, rows)
    ax = Axis(fig[1, j]; title=dimlabel(rd[1]) * ", tol 1e-8", ylabel=j == 1 ? "time / ms" : "")
    gs = filter(g -> any(r -> label(r) == g, rd), groups)
    for (i, g) in enumerate(gs)
      r = first(filter(r -> label(r) == g, rd))
      barplot!(ax, [i - 0.2, i + 0.2], [ms(r["t_trafo"]), ms(r["t_adj"])]; width=0.38,
               color=[gcolor[g], (gcolor[g], 0.5)])
    end
    ax.xticks = (1:length(gs), gs); ax.xticklabelrotation = π / 5
  end
  Legend(fig[2, :], [PolyElement(color=:gray40), PolyElement(color=(:gray40, 0.5))], ["direct", "adjoint/transposed"];
         orientation=:horizontal, framevisible=false)
  save_fig("real", fig)
end

# all measurements for the explorer of the documentation
jsonval(v::AbstractString) = "\"" * replace(v, "\\" => "\\\\", "\"" => "\\\"") * "\""
jsonval(v::AbstractVector) = "[" * join(jsonval.(v), ",") * "]"
jsonval(v::Real) = isfinite(v) ? string(v) : "null"
jsonval(v) = jsonval(string(v))
const FIELDS = ["suite", "lib", "transform", "D", "N", "J", "L", "nodes", "T", "threads", "tol", "err_trafo",
                "err_adj", "t_plan", "t_trafo", "t_adj", "W", "sigma"]
function data()
  rows = Dict{String,Any}[]; metas = String[]
  for f in sort(filter(endswith(".csv"), readdir(SRC)))
    meta, rs = read_csv(joinpath(SRC, f))
    append!(rows, filter(ok, rs))
    push!(metas, jsonval(first(f, length(f) - 4)) * ":{" *
                 join([jsonval(k) * ":" * jsonval(v) for (k, v) in meta if k != "command"], ",") * "}")
  end
  open(joinpath(OUT, "data.js"), "w") do io
    print(io, "window.NFFT_BENCH = {meta:{", join(metas, ","), "},\nrows:[\n")
    for (i, r) in enumerate(rows)
      print(io, "{", join([jsonval(c) * ":" * jsonval(get(r, c, "")) for c in FIELDS], ","), "}",
            i < length(rows) ? ",\n" : "\n")
    end
    print(io, "]};\n")
  end
  println("  data.js ($(length(rows)) measurements)")
end

println("report from $SRC:")
accuracy(); density(); sizes(); threads(); batch(); real(); accuracy(; suite="nnfft", name="nnfft")
data()
