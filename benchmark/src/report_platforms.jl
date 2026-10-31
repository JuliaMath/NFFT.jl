# Figures across platforms, included by report.jl: overview of the speed ratios (CPU, CUDA, Metal),
# time against measured error on the GPUs, NFFT.jl on four devices, delivered error against the
# requested tolerance. Machines without a results folder are left out.

const PLATFORMS = (
  m1    = (machine="apple-m1-max", label="M1 Max CPU"),
  zen3  = (machine="amd-epyc-7763", label="Zen 3 CPU"),
  metal = (machine="apple-m1-max", label="M1 Max GPU (Metal)"),
  cuda  = (machine="rtx-a6000", label="RTX A6000 (CUDA)"),
)
const CPU_MACHINES = ["apple-m1-max", "amd-epyc-7763"]
const STANDARD = Dict(1 => [2^18], 2 => [512, 512], 3 => [64, 64, 64])
const BIG = Dict(1 => [2^22], 2 => [2048, 2048], 3 => [160, 160, 160])
const DISPLAY = Dict("NFFT" => "NFFT.jl", "NonuniformFFTs" => "NonuniformFFTs.jl", "NFFT 0.14" => "NFFT.jl 0.14")
display_name(lib) = lib == "cuFINUFFT" ? lib : get(DISPLAY, family(lib), family(lib))
isref(r) = family(r["lib"]) == "NFFT" && r["lib"] != "NFFT 0.14"

# cuFINUFFT writes its own CSV (benchmark/cufinufft_bench.py); its rows get the fields of the suite `cuda`
function load_cufinufft(machine)
  rows = Dict{String,Any}[]
  for f in ("cufinufft.csv", "cufinufft-big.csv")
    file = joinpath(RESULTS, machine, f)
    isfile(file) || continue
    for r in read_csv(file)[2]
      r["err_trafo"] isa Real || continue
      push!(rows, merge(r, Dict{String,Any}("suite" => "cuda", "transform" => "nfft", "nodes" => "uniform", "L" => 1)))
    end
  end
  rows
end

function platform_rows(p)
  m = PLATFORMS[p].machine
  isdir(joinpath(RESULTS, m)) || return Dict{String,Any}[]
  p == :cuda ? [load("cuda"; machine=m)[2]; load_cufinufft(m)] :
  p == :metal ? load("metal"; machine=m)[2] : load("pareto"; machine=m)[2]
end

valid(r, err, t) = r[err] isa Real && r[t] isa Real && r[err] > 0 && isfinite(r[t])

# Pareto front: the points no other point beats in both error and time, i.e. the fastest time of
# a library for each error it reaches, over all its tolerances and settings. Errors within 15 % of
# each other count as equal (the faster point stays): a library whose error stalls returns nearly
# the same error for many tolerances, which would otherwise draw vertical steps.
function front(rows, err, t)
  out = eltype(rows)[]; best = Inf
  for r in sort(filter(r -> valid(r, err, t), rows); by=r -> (r[err], r[t]))
    r[t] < best || continue
    !isempty(out) && r[err] < 1.15 * out[end][err] && pop!(out)
    push!(out, r); best = r[t]
  end
  out
end

# (error, time of the library / time of NFFT.jl at that error) on the Pareto fronts of both
ratios(ref, other, err, t) =
  [(r[err], r[t] / tn) for r in front(other, err, t) for tn in (time_at(front(ref, err, t), r[err], err, t),)
   if tn !== nothing]

# 1–2–5 ticks, denser (1–2–3–5–7, 1–1.5–2–3–5–7) for narrow ranges
function time_ticks(lo, hi)
  for ms in ([1, 2, 5], [1, 2, 3, 5, 7], [1, 1.5, 2, 3, 5, 7], [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9])
    ts = filter(v -> lo <= v <= hi, [m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in ms])
    length(ts) >= 3 && return ts
  end
  val_ticks(lo, hi)
end
ratio_ticks(lo, hi) = filter(v -> lo / 1.05 <= v <= hi * 1.05, [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100])

# A. Median speed ratio per competitor and platform (rows) and dimension and transform (columns),
# Pareto fronts of both libraries, standard sizes (1D 2^18, 2D 512², 3D 64³, J = |N|).
const OVERVIEW_ROWS = [("ducc", :m1), ("ducc", :zen3), ("FINUFFT", :m1), ("FINUFFT", :zen3),
                       ("NonuniformFFTs", :m1), ("NonuniformFFTs", :zen3), ("NonuniformFFTs", :cuda),
                       ("cuFINUFFT", :cuda), ("mlx-nufft", :metal)]

function overview()
  data = Dict(p => filter(r -> r["N"] == STANDARD[r["D"]] && r["J"] == prod(r["N"]), platform_rows(p))
              for p in keys(PLATFORMS))
  libmatch(r, lib) = lib == "cuFINUFFT" ? r["lib"] == lib : family(r["lib"]) == lib && r["lib"] != "cuFINUFFT"
  cols = [(D, op...) for D in 1:3 for op in OPS]
  panels = []
  for T in ("Float64", "Float32")
    rows = []
    for (lib, p) in OVERVIEW_ROWS
      rd = filter(r -> r["T"] == T, data[p])
      any(r -> libmatch(r, lib), rd) || continue
      cells = map(cols) do (D, _, err, t)
        other = filter(r -> r["D"] == D && libmatch(r, lib), rd)
        isempty(other) && return :missing
        rs = last.(ratios(filter(r -> r["D"] == D && isref(r), rd), other, err, t))
        isempty(rs) ? :nooverlap : (median(rs), extrema(rs)..., length(rs))
      end
      push!(rows, ("$(display_name(lib)) · $(PLATFORMS[p].label)", cells))
    end
    isempty(rows) || push!(panels, (T, rows))
  end
  isempty(panels) && return

  cmap = cgrad(:RdBu); lim = 2.0                 # colours saturate at 1/4 and 4
  hatch = Makie.LinePattern(direction=Vec2f(1, 1), width=1.2, tilesize=(7, 7), linecolor=:gray55, backgroundcolor=:gray96)
  rowh = 38
  fig = Figure(size=(900, 120 + sum(rowh * length(p[2]) + 70 for p in panels)))
  println("  overview (median, min, max of time / time of NFFT.jl):")
  for (k, (T, rows)) in enumerate(panels)
    n = length(rows)
    ax = Axis(fig[k, 1]; yreversed=true, xaxisposition=:top, limits=((0.5, 6.5), (0.5, n + 0.5)),
              xticks=(1:6, ["$(D)D $(op)" for (D, op) in cols]), yticks=(1:n, first.(rows)), xticklabelsize=14, yticklabelsize=14,
              title="$(T == "Float64" ? "double" : "single") precision ($T)", titlealign=:left,
              xgridvisible=false, ygridvisible=false, xticksvisible=false, yticksvisible=false,
              leftspinevisible=false, rightspinevisible=false, topspinevisible=false, bottomspinevisible=false)
    rowsize!(fig.layout, k, Fixed(rowh * n + 10))
    for (i, (label, cells)) in enumerate(rows), (j, c) in enumerate(cells)
      box = Rect2f(j - 0.48, i - 0.46, 0.96, 0.92)
      if c === :missing
        poly!(ax, box; color=:white, strokecolor=:gray80, strokewidth=0.5)
        text!(ax, j, i; text="no data", align=(:center, :center), fontsize=11, color=:gray50)
      elseif c === :nooverlap
        poly!(ax, box; color=hatch, strokecolor=:gray80, strokewidth=0.5)
        text!(ax, j, i; text="no overlap", align=(:center, :center), fontsize=11, color=:black, font=:bold)
      else
        med, lo, hi, npts = c
        v = clamp(log2(med), -lim, lim)
        poly!(ax, box; color=cmap[(v + lim) / 2lim])
        tc = abs(v) > 1.7 ? :white : :black
        text!(ax, j, i - 0.12; text=fmt_ratio(med), align=(:center, :center), fontsize=16, font=:bold, color=tc)
        text!(ax, j, i + 0.24; text=npts == 1 ? "1 point" : "$(fmt_ratio(lo))–$(fmt_ratio(hi))", align=(:center, :center),
              fontsize=12, color=tc)
      end
      println("    ", rpad(T, 8), rpad(label, 46), rpad(cols[j][1], 2), rpad(cols[j][2], 8),
              c isa Symbol ? c : join(fmt_ratio.(c[1:3]), " ") * " ($(c[4]))")
    end
  end
  Colorbar(fig[length(panels) + 1, 1]; colormap=cmap, limits=(-lim, lim), vertical=false, flipaxis=false,
           ticks=(-2:2, ["≤ ¼", "½", "1", "2", "≥ 4"]), width=Relative(0.5),
           label="time of the library / time of NFFT.jl at equal error (> 1: NFFT.jl faster)")
  save_fig("overview", fig)
end
median(x) = (s = sort(x); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
pow10s(v) = string(round(v; sigdigits=2))
fmt_ratio(v) = v >= 10 ? string(round(Int, v)) : string(round(v; digits=v >= 1 ? 1 : 2))

# B. Time against measured error on the GPU: the Pareto front of each library, standard or big sizes
function pareto_time(name, rows, sizes; libs, title)
  rows = filter(r -> r["N"] == sizes[r["D"]], rows)
  isempty(rows) && return
  Ts = filter(T -> any(r -> r["T"] == T, rows), ["Float64", "Float32"])
  fig = Figure(size=(900, 560))
  for (j, D) in enumerate(1:3), (i, (op, err, t)) in enumerate(OPS)
    rd = filter(r -> r["D"] == D, rows)
    ax = Axis(fig[i, j]; LOGX_EXP..., yscale=log10, yticks=time_ticks, ytickformat=val_format, xreversed=true,
              title=i == 1 ? dimlabel(rd[1]) : "",
              xlabel=i == 2 ? "relative error" : "", ylabel=j == 1 ? "$op / ms" : "")
    for lib in libs, T in Ts
      f = front(filter(r -> r["lib"] == lib && r["T"] == T, rd), err, t)
      isempty(f) && continue
      scatterlines!(ax, [r[err] for r in f], [ms(r[t]) for r in f]; color=color(lib),
                    linestyle=T == "Float32" ? :dash : :solid, markersize=6)
    end
  end
  Label(fig[0, :], title; fontsize=14, halign=:left, tellwidth=false)
  els = [[LineElement(color=color(l), linewidth=2), MarkerElement(color=color(l), marker=:circle)] for l in libs]
  styles = length(Ts) == 2 ? [[LineElement(color=:black)], [LineElement(color=:black, linestyle=:dash)]] : []
  Legend(fig[3, :], [els; styles], [display_name.(libs); length(Ts) == 2 ? Ts : String[]];
         orientation=:horizontal, framevisible=false)
  save_fig(name, fig)
end

# The speed ratios on CUDA (Pareto fronts of both libraries), Float64 solid, Float32 dashed
function gpu_ratio(name, rows, sizes; libs, title)
  rows = filter(r -> r["N"] == sizes[r["D"]], rows)
  isempty(rows) && return
  fig = Figure(size=(900, 580)); axes = Axis[]; ys = Float64[]
  for (j, D) in enumerate(1:3), (i, (op, err, t)) in enumerate(OPS)
    rd = filter(r -> r["D"] == D, rows)
    ax = Axis(fig[i, j]; LOGX_EXP..., yscale=log10, xreversed=true, title=i == 1 ? dimlabel(rd[1]) : "",
              xlabel=i == 2 ? "relative error" : "", ylabel=j == 1 ? "$op: time / time of NFFT.jl" : "")
    push!(axes, ax); hlines!(ax, [1]; color=(:black, 0.4), linewidth=1)
    for lib in libs, T in ("Float64", "Float32")
      pts = ratios(filter(r -> isref(r) && r["T"] == T, rd), filter(r -> r["lib"] == lib && r["T"] == T, rd), err, t)
      isempty(pts) && continue
      append!(ys, last.(pts))
      println("    $name $(D)D ", rpad(op, 8), rpad(display_name(lib), 18), rpad(T, 8),
              join(fmt_ratio.(extrema(last.(pts))), "–"), " at ", join(pow10s.(extrema(first.(pts))), "…"))
      scatterlines!(ax, first.(pts), last.(pts); color=color(lib), linestyle=T == "Float32" ? :dash : :solid, markersize=6)
    end
  end
  linkyaxes!(axes...)
  ticks = ratio_ticks(extrema(ys)...)
  for ax in axes
    ax.yticks = ticks; ax.ytickformat = val_format
  end
  Label(fig[0, :], title; fontsize=14, halign=:left, tellwidth=false)
  els = [[LineElement(color=color(l), linewidth=2), MarkerElement(color=color(l), marker=:circle)] for l in libs]
  Legend(fig[3, :], [els; first.(PRECISION_STYLES)], [display_name.(libs); last.(PRECISION_STYLES)];
         orientation=:horizontal, framevisible=false)
  save_fig(name, fig)
end

function gpus()
  cuda = platform_rows(:cuda)
  culibs = ["NFFT CUDA", "cuFINUFFT", "NonuniformFFTs CUDA"]
  pareto_time("cuda-time", cuda, STANDARD; libs=culibs, title="RTX A6000 (CUDA), J = |N|")
  pareto_time("cuda-time-big", cuda, BIG; libs=culibs, title="RTX A6000 (CUDA), large sizes, J = |N|")
  gpu_ratio("cuda-ratio", cuda, STANDARD; libs=culibs[2:3], title="RTX A6000 (CUDA), J = |N|")
  gpu_ratio("cuda-ratio-big", cuda, BIG; libs=culibs[2:3], title="RTX A6000 (CUDA), large sizes, J = |N|")
  pareto_time("metal-time", platform_rows(:metal), STANDARD; libs=["NFFT Metal", "mlx-nufft"],
              title="M1 Max GPU (Metal), single precision (Float32), J = |N|")
end

# C. NFFT.jl's time for the standard cases on every device: Float64 at tolerance 1e-8 and Float32 at
# 1e-5 (Metal has no Float64)
function devices()
  devs = [(:m1, "M1 Max\nCPU"), (:zen3, "Zen 3\nCPU"), (:metal, "M1 Max\nGPU"), (:cuda, "A6000\nGPU")]
  rows = Dict(p => filter(r -> isref(r) && r["N"] == STANDARD[r["D"]] && r["J"] == prod(r["N"]), platform_rows(p))
              for (p, _) in devs)
  precs = [("Float64", 1e-8, COLORS["NFFT"]), ("Float32", 1e-5, "#8fb0e3")]
  fig = Figure(size=(900, 520))
  println("  devices (ms, direct / adjoint):")
  for (j, D) in enumerate(1:3), (i, (op, _, t)) in enumerate(OPS)
    ax = Axis(fig[i, j]; title=i == 1 ? "$(D)D, N = $(D == 1 ? "2¹⁸" : D == 2 ? "512²" : "64³")" : "",
              ylabel=j == 1 ? "$op / ms" : "", xticks=(1:length(devs), last.(devs)),
              xticklabelsvisible=i == 2, xgridvisible=false, xticklabelsize=11)
    top = 0.0
    for (k, (p, _)) in enumerate(devs), (q, (T, tol, c)) in enumerate(precs)
      x = k + (q - 1.5) * 0.4
      r = filter(r -> r["D"] == D && r["T"] == T && isapprox(r["tol"], tol; rtol=1e-6), rows[p])
      if isempty(r)
        text!(ax, x, 0; text="n/a", align=(:center, :bottom), fontsize=10, color=:gray45)
        continue
      end
      v = ms(only(r)[t]); top = max(top, v)
      barplot!(ax, [x], [v]; width=0.38, color=c)
      text!(ax, x, v; text=val_format([round(v; sigdigits=2)])[1], align=(:center, :bottom), fontsize=10, offset=(0, 2))
      i == 1 && println("    ", rpad(PLATFORMS[p].label, 22), "$(D)D ", rpad(T, 8), " ",
                        round(v; sigdigits=3), " / ", round(ms(only(r)["t_adj"]); sigdigits=3))
    end
    ylims!(ax, 0, 1.15top)
  end
  Legend(fig[3, :], [PolyElement(color=c) for (_, _, c) in precs], ["Float64, tolerance 10⁻⁸", "Float32, tolerance 10⁻⁵"];
         orientation=:horizontal, framevisible=false)
  save_fig("devices", fig)
end

# D. Delivered error against the requested tolerance (suite `contract`): the largest error over
# the dimensions and over direct and adjoint transform, per library and precision
function contract()
  _, rows = load("contract"; machine=MACHINE)
  isempty(rows) && return
  libs = filter(l -> any(r -> r["lib"] == l, rows), ["NFFT", "FINUFFT", "ducc", "NonuniformFFTs", "NFFT 0.14"])
  fig = Figure(size=(900, 420)); axes = Axis[]
  for (j, (nodes, title)) in enumerate((("uniform", "uniformly random nodes"), ("cartesian", "nodes on a regular lattice")))
    rn = filter(r -> r["nodes"] == nodes && r["err_trafo"] isa Real, rows)
    ax = Axis(fig[1, j]; LOGX_EXP..., yscale=log10, yticks=exp_ticks, ytickformat=exp_format, xreversed=true,
              title, xlabel="requested tolerance", ylabel=j == 1 ? "delivered error (largest over D, direct, adjoint)" : "")
    push!(axes, ax)
    tols = sort(unique(r["tol"] for r in rn))
    lines!(ax, tols, tols; color=(:black, 0.5), linestyle=:dot, linewidth=1.5)
    for lib in libs, T in ("Float64", "Float32")
      rl = filter(r -> r["lib"] == lib && r["T"] == T, rn)
      isempty(rl) && continue
      ts = sort(unique(r["tol"] for r in rl))
      es = [maximum(max(r["err_trafo"], r["err_adj"]) for r in rl if r["tol"] == tol) for tol in ts]
      scatterlines!(ax, ts, es; color=color(lib), linestyle=T == "Float32" ? :dash : :solid,
                    marker=T == "Float32" ? :utriangle : :circle, markersize=7)
    end
  end
  linkaxes!(axes...)
  els = [[LineElement(color=color(l), linewidth=2), MarkerElement(color=color(l), marker=:circle)] for l in libs]
  Legend(fig[2, :], [els; [[LineElement(color=(:black, 0.5), linestyle=:dot)]]; first.(PRECISION_STYLES)],
         [display_name.(libs); "error = tolerance"; last.(PRECISION_STYLES)]; orientation=:horizontal, framevisible=false,
         nbanks=2)
  save_fig("contract", fig)
end
