"""
Benchmark suite of NFFT.jl: orchestrates worker processes, one per library and thread count, each
in the library's own environment (`envs/`), and collects their results into CSV files.
Standard library only, so that it runs without a project.
"""
module Benchmarks

using TOML, Printf, Dates, Random

const ROOT = dirname(@__DIR__)                   # benchmark/
const REPO = dirname(ROOT)                       # NFFT.jl repository
const CACHE = joinpath(ROOT, ".cache")

include("suites.jl")

# Libraries: environment and adapter. "NFFT@<rev>" is NFFT.jl at a git revision (see `ref_lib`).
const LIBS = Dict(
  "NFFT" => (env=joinpath(ROOT, "envs", "nfft"), adapter="nfft.jl"),
  "NFFT 0.14" => (env=joinpath(ROOT, "envs", "nfft014"), adapter="nfft014.jl"),
  "FINUFFT" => (env=joinpath(ROOT, "envs", "finufft"), adapter="finufft.jl"),
  "ducc" => (env=joinpath(ROOT, "envs", "ducc"), adapter="ducc.jl"),
  "NonuniformFFTs" => (env=joinpath(ROOT, "envs", "nonuniformffts"), adapter="nonuniformffts.jl"),
  "NFFT Metal" => (env=joinpath(ROOT, "envs", "nfftmetal"), adapter="nfft_metal.jl"),
  "mlx-nufft" => (env=joinpath(ROOT, "envs", "mlxnufft"), adapter="mlxnufft.jl"),
)

const COLUMNS = ["suite", "lib", "transform", "D", "N", "J", "L", "nodes", "T", "threads", "tol", "sigma_req", "round", "load",
                 "err_trafo", "err_adj", "t_plan", "t_trafo", "t_adj", "t_trafo_med", "t_adj_med",
                 "alloc_trafo", "alloc_adj", "W", "sigma", "window", "lib_version", "lib_library",
                 "skipped", "error"]

julia() = `$(Base.julia_cmd()) --startup-file=no`

function libspec(name)
  startswith(name, "NFFT@") && return ref_lib(name[6:end])
  haskey(LIBS, name) || error("unknown library $name; known: $(join(sort(collect(keys(LIBS))), ", ")), NFFT@<rev>")
  LIBS[name]
end

# Instantiate an environment once (its Manifest.toml is not versioned).
function ensure_env(env)
  isfile(joinpath(env, "Manifest.toml")) && return
  @info "instantiating $env"
  run(`$(julia()) --project=$env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`)
end

# NFFT.jl at a git revision: a worktree of this repository and an environment developing it.
function ref_lib(rev)
  sha = readchomp(`git -C $REPO rev-parse --short $rev`)
  dir = joinpath(CACHE, "rev-$sha")
  src = joinpath(dir, "src")
  if !isdir(src)
    mkpath(dir)
    run(`git -C $REPO worktree add --detach --force $src $sha`)
  end
  env = joinpath(dir, "env")
  # AbstractNFFTs is in lib/, in older revisions at the top level
  abs = isdir(joinpath(src, "lib", "AbstractNFFTs")) ? joinpath(src, "lib", "AbstractNFFTs") : joinpath(src, "AbstractNFFTs")
  if !isfile(joinpath(env, "Project.toml"))
    mkpath(env)
    write(joinpath(env, "Project.toml"), """
    [deps]
    AbstractNFFTs = "7f219486-4aa7-41d6-80a7-e08ef20ceed7"
    NFFT = "efe261a4-0d2b-5849-be55-fc731d526b0d"

    [sources]
    AbstractNFFTs = {path = "$abs"}
    NFFT = {path = "$src"}
    """)
  end
  (env=env, adapter="nfft.jl")
end

"""
    run_tasks(tasks; io=stdout) -> Vector{Dict}

Run the tasks, grouped by library and thread count, each group in one worker process.
"""
function run_tasks(tasks; io=stdout)
  groups = Dict{Tuple{String,Int},Vector{Dict{String,Any}}}()
  order = Tuple{String,Int}[]
  for t in tasks
    key = (t["lib"], t["threads"])
    haskey(groups, key) || push!(order, key)
    push!(get!(groups, key, Dict{String,Any}[]), t)
  end
  results = Dict{String,Any}[]
  for key in order
    append!(results, run_group(key..., groups[key]; io))
  end
  results
end

# `rounds` repetitions of all tasks, each round in a new random order (seed = round): the cases (tasks
# on the same problem) in random order and the tasks of a case in random order, so that drift
# (thermal, background load) spreads over the candidates. Every result records its round and the
# load average (1, 5, 15 min) when the round started.
case_key(t) = (t["transform"], t["N"], t["J"], get(t, "L", 1), t["nodes"], t["T"], t["tol"], get(t, "seed", 1234))
function run_rounds(tasks, rounds; io=stdout)
  results = Dict{String,Any}[]
  for r in 1:rounds
    rng = Random.Xoshiro(r)
    cases = Dict{Any,Vector{Dict{String,Any}}}(); order = Any[]
    for t in tasks
      k = case_key(t)
      haskey(cases, k) || push!(order, k)
      push!(get!(cases, k, Dict{String,Any}[]), t)
    end
    load = loadavg()
    println(io, "═══ round $r of $rounds (load $load)")
    shuffled = reduce(vcat, [Random.shuffle(rng, cases[k]) for k in Random.shuffle(rng, order)])
    append!(results, run_tasks([merge(t, Dict{String,Any}("round" => r, "load" => load)) for t in shuffled]; io))
  end
  results
end

function run_group(lib, threads, tasks; io=stdout)
  spec = libspec(lib)
  ensure_env(spec.env)
  mkpath(CACHE)
  tfile, rfile = joinpath(CACHE, "tasks.toml"), joinpath(CACHE, "results.toml")
  rm(rfile; force=true)
  open(io -> TOML.print(io, Dict("adapter" => joinpath(ROOT, "worker", "adapters", spec.adapter),
                                 "task" => tasks)), tfile, "w")
  println(io, "── $lib, $threads thread$(threads == 1 ? "" : "s"): $(length(tasks)) task$(length(tasks) == 1 ? "" : "s")")
  cmd = `$(julia()) --project=$(spec.env) --threads=$threads $(joinpath(ROOT, "worker", "worker.jl")) $tfile $rfile`
  ok = success(pipeline(cmd; stdout=io, stderr=stderr))
  res = isfile(rfile) ? TOML.parsefile(rfile)["result"] : Dict{String,Any}[]
  if !ok
    println(io, "   worker failed after $(length(res)) of $(length(tasks)) tasks")
    for t in tasks[length(res)+1:end]
      push!(res, merge(t, Dict("error" => "worker process failed")))
    end
  end
  res
end

# ── results ──────────────────────────────────────────────────────────────────────────────────

function machine()
  name = try
    Sys.isapple() ? readchomp(`sysctl -n machdep.cpu.brand_string`) :
      strip(split(first(filter(startswith("model name"), readlines("/proc/cpuinfo"))), ":")[2])
  catch
    Sys.CPU_NAME
  end
  lowercase(replace(strip(name), r"[^A-Za-z0-9]+" => "-"))
end

function repo_commit()
  sha = readchomp(`git -C $REPO rev-parse --short HEAD`)
  dirty = !isempty(readchomp(`git -C $REPO status --porcelain --untracked-files=no`))
  dirty ? sha * "+" : sha
end

csvcell(v::AbstractVector) = join(v, "x")
csvcell(v::AbstractFloat) = isnan(v) ? "" : repr(v)
csvcell(v::AbstractString) = occursin(r"[,\"\n]", v) ? "\"" * replace(v, "\"" => "\"\"") * "\"" : v
csvcell(v) = string(v)

function row(r)
  r = copy(r)
  r["D"] = length(r["N"])
  [csvcell(get(r, c, "")) for c in COLUMNS]
end

function write_csv(file, results; meta=Dict{String,Any}())
  mkpath(dirname(file))
  open(file, "w") do io
    for (k, v) in sort(collect(meta); by=first)
      println(io, "# ", k, ": ", v)
    end
    println(io, join(COLUMNS, ","))
    for r in results
      println(io, join(row(r), ","))
    end
  end
  file
end

function parse_csv_line(line)
  cells = String[]; buf = IOBuffer(); q = false; i = 1
  cs = collect(line)
  while i <= length(cs)
    c = cs[i]
    if q
      if c == '"' && i < length(cs) && cs[i+1] == '"'
        write(buf, '"'); i += 1
      elseif c == '"'
        q = false
      else
        write(buf, c)
      end
    elseif c == '"'
      q = true
    elseif c == ','
      push!(cells, String(take!(buf)))
    else
      write(buf, c)
    end
    i += 1
  end
  push!(cells, String(take!(buf)))
  cells
end

"""
    read_csv(file) -> (meta, rows)

Rows as Dicts with numbers parsed; `N` as a vector of Ints.
"""
function read_csv(file)
  meta = Dict{String,String}(); rows = Dict{String,Any}[]; cols = String[]
  for line in eachline(file)
    if startswith(line, "# ")
      k, v = split(line[3:end], ": "; limit=2); meta[k] = v
    elseif isempty(cols)
      cols = split(line, ",")
    else
      cells = parse_csv_line(line)
      r = Dict{String,Any}()
      for (c, v) in zip(cols, cells)
        r[c] = c == "N" ? parse.(Int, split(v, "x")) : something(tryparse(Int, v), tryparse(Float64, v), v)
      end
      push!(rows, r)
    end
  end
  meta, rows
end

# Load average (1, 5, 15 min) when the orchestrator started and when the results are written.
const LOAD_START = Ref("")
loadavg() = join(round.(Sys.loadavg(); digits=2), " ")

meta_info(suite, args) = Dict{String,Any}("date" => string(now()), "machine" => machine(),
  "cpu" => Sys.CPU_NAME, "julia" => string(VERSION), "commit" => repo_commit(),
  "suite" => suite, "command" => join(args, " "), "load_start" => LOAD_START[], "load_end" => loadavg())

# ── comparison ───────────────────────────────────────────────────────────────────────────────

key(r; lib=true) = (lib ? r["lib"] : "", r["transform"], r["N"], r["J"], get(r, "L", 1), r["nodes"], r["T"],
                    r["threads"], r["tol"], get(r, "sigma_req", ""), get(r, "round", ""))
ok(r) = get(r, "skipped", "") == "" && get(r, "error", "") == ""
geomean(x) = isempty(x) ? NaN : exp(sum(log, x) / length(x))

"""
    compare(new, old; io)

Table of time ratios old/new (> 1: new is faster) for matching measurements of two result lists
(matched by everything but the library name if `bylib = false`).
"""
function compare(new, old; io=stdout, bylib=true, labels=("new", "old"))
  idx = Dict(key(r; lib=bylib) => r for r in old if ok(r))
  @printf(io, "%-14s %-5s %-12s %-5s %-7s %2s %7s | %-17s %-17s | %6s %6s | %8s %8s\n", "lib", "kind", "N", "L",
          "T", "th", "tol", "trafo $(labels[1])/$(labels[2])", "adj $(labels[1])/$(labels[2])",
          "×trafo", "×adj", "err new", "err old")
  rt, ra = Float64[], Float64[]
  for r in new
    ok(r) || continue
    o = get(idx, key(r; lib=bylib), nothing)
    o === nothing && continue
    a, b = o["t_trafo"] / r["t_trafo"], o["t_adj"] / r["t_adj"]
    push!(rt, a); push!(ra, b)
    @printf(io, "%-14s %-5s %-12s %-5d %-7s %2d %7.0e | %7.2f / %7.2f %7.2f / %7.2f | %6.2f %6.2f | %8.1e %8.1e\n",
            r["lib"], r["transform"], join(r["N"], "x"), get(r, "L", 1), r["T"], r["threads"], r["tol"],
            1e3r["t_trafo"], 1e3o["t_trafo"], 1e3r["t_adj"], 1e3o["t_adj"], a, b, r["err_trafo"], o["err_trafo"])
  end
  @printf(io, "geometric mean of the speedups (> 1: %s faster): trafo %.3f, adjoint %.3f (%d measurements)\n",
          labels[1], geomean(rt), geomean(ra), length(rt))
  (trafo=geomean(rt), adjoint=geomean(ra))
end

# Fastest of several rounds per measurement.
function best_of(rounds)
  best = Dict{Any,Dict{String,Any}}()
  order = Any[]
  for res in rounds, r in res
    ok(r) || continue
    k = key(r)
    if !haskey(best, k)
      best[k] = copy(r); push!(order, k)
    else
      for c in ("t_trafo", "t_adj", "t_plan")
        best[k][c] = min(best[k][c], r[c])
      end
    end
  end
  [best[k] for k in order]
end

"""
    ab(rev, tasks; rounds=2)

NFFT.jl of the working tree against revision `rev`: both run the same tasks, alternating for
`rounds` rounds to average out drift (thermal, background load); the fastest of the rounds counts.
"""
function ab(rev, tasks; rounds=2, io=stdout)
  revlib = "NFFT@" * rev
  cur = filter(t -> t["lib"] == "NFFT", tasks)
  old = [merge(t, Dict("lib" => revlib)) for t in cur]
  rc, ro = [], []
  for i in 1:rounds
    println(io, "═══ round $i of $rounds")
    push!(ro, run_tasks(old; io)); push!(rc, run_tasks(cur; io))
  end
  bc, bo = best_of(rc), best_of(ro)
  println(io)
  compare(bc, [merge(r, Dict("lib" => "NFFT")) for r in bo]; io, labels=("now", rev))
  bc, bo
end

end
