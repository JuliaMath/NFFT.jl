# Benchmark worker: runs the tasks of one library in the library's own environment.
#
#   julia --project=<env> -t <threads> worker.jl <tasks.toml> <results.toml>
#
# tasks.toml holds `adapter` (path of the library adapter) and an array `task`. Each result is the
# task plus the measured values; the results file is rewritten after every task, so that a crash
# keeps what was measured.

using TOML, Printf, Statistics
include("problems.jl")

# thrown by an adapter for a problem the library cannot do; recorded as skipped
struct Unsupported <: Exception
  reason::String
end

# What a C library prints to stdout while f runs (the debug output of FINUFFT, ducc's verbose report).
function capture_c(f)
  path, io = mktemp()
  r = try
    redirect_stdout(io) do
      x = f(); Libc.flush_cstdio(); x
    end
  finally
    close(io)
  end
  out = read(path, String); rm(path)
  r, out
end

include("fftw.jl")
include(joinpath(@__DIR__, "..", "src", "rules.jl"))
using .Rules

# the tolerance the library actually got, where its adapter raises or clamps the requested one
const TOL_USED = Ref(NaN)

const CFG = TOML.parsefile(ARGS[1])
include(CFG["adapter"])

# defaults of the adapter interface
@isdefined(release!) || (release!(p) = nothing)
@isdefined(params) || (params(p) = Dict{String,Any}())
@isdefined(stages) || (stages(p, prob) = Dict{String,Any}())
# GPU libraries: data moved to the device before the measurements, results back for the error, and
# a synchronisation inside every timed call
@isdefined(upload) || (upload(x) = x)
@isdefined(tohost) || (tohost(x) = x)
@isdefined(sync) || (sync() = nothing)

# Minimum and median wall time: one warm-up call, then repeated calls until `budget` seconds have
# passed (at least 3, at most 1000).
function timeit(f, budget)
  f(); sync()
  ts = Float64[]
  t0 = time()
  while length(ts) < 3 || (time() - t0 < budget && length(ts) < 1000)
    push!(ts, @elapsed (f(); sync()))
  end
  minimum(ts), median(ts)
end

problem_key(t) = (t["transform"], t["N"], t["J"], get(t, "L", 1), t["nodes"], t["T"], get(t, "seed", 1234))

function natural(prob::Problem)
  L = size(prob.y, 2)
  L == 1 ? (reshape(prob.x, Base.front(size(prob.x))), vec(prob.y)) : (prob.x, prob.y)
end

function measure(prob::Problem, task)
  # the cached problem carries the task that created it; the plan needs this one (e.g. its sigma_req)
  prob = Problem((f === :task ? task : getfield(prob, f) for f in fieldnames(Problem))...)
  tol = Float64(task["tol"]); budget = Float64(get(task, "budget", 0.1))
  x, y = map(upload, natural(prob))
  # FFTW_MEASURE plans take up to minutes for large grids: only the timed plan uses the task's flag;
  # the compiling plan and the reported grid (independent of the flag) use ESTIMATE
  est = Problem((f === :task ? merge(task, Dict("fft_flag" => "ESTIMATE")) : getfield(prob, f)
                 for f in fieldnames(Problem))...)
  forget_fftw_wisdom()
  p = plan(est, tol)                                   # compiles
  release!(p)
  t_plan = Inf
  for _ in 1:Int(get(task, "plans", 3)) - 1
    forget_fftw_wisdom()
    t_plan = min(t_plan, @elapsed(q = plan(prob, tol)))
    release!(q)
  end
  TOL_USED[] = NaN
  forget_fftw_wisdom()
  t_plan = min(t_plan, @elapsed(p = plan(prob, tol)))
  tol_used = isnan(TOL_USED[]) ? tol : TOL_USED[]
  outT = similar(y); outA = similar(x)
  t_trafo, t_trafo_med = timeit(() -> trafo!(p, outT, x), budget)
  t_adj, t_adj_med = timeit(() -> adjoint!(p, outA, y), budget)
  r = Dict{String,Any}(
    "err_trafo" => err_trafo(prob, tohost(outT)), "err_adj" => err_adjoint(prob, tohost(outA)),
    "tol_used" => tol_used, "t_plan" => t_plan, "t_trafo" => t_trafo, "t_adj" => t_adj,
    "t_trafo_med" => t_trafo_med, "t_adj_med" => t_adj_med,
    "alloc_trafo" => @allocated(trafo!(p, outT, x)), "alloc_adj" => @allocated(adjoint!(p, outA, y)))
  merge!(r, params(p))
  @isdefined(fft_flag) && (r["fft_flag"] = fft_flag(task))
  get(task, "stages", false) && merge!(r, stages(p, prob))
  release!(p)
  # the grid Ñ per dimension: from the plan, or what the library reports (outside the timing)
  need = !haskey(r, "rep_W") && (haskey(task, "rule") || !haskey(r, "grid"))
  rep = need && @isdefined(reported) ? reported(est, tol) : r
  haskey(rep, "rep_grid") && !haskey(r, "grid") && (r["grid"] = rep["rep_grid"]; r["grid_adj"] = rep["rep_grid_adj"])
  haskey(r, "grid") && !haskey(r, "grid_adj") && (r["grid_adj"] = r["grid"])
  haskey(task, "rule") && merge!(r, rule_check(prob, task, tol, rep))
  r
end

# Suite `decomposition`: the parameters of the rule (src/rules.jl) next to those the library reports
# (rep_*: NFFT.jl with the rule's parameters, or the original library's own choice, in `rep`), and
# the fields where they differ. σ and β of FINUFFT are printed with three digits, hence the relative 1 %.
function rule_check(prob, task, tol, rep)
  c = Rules.choose(task["rule"], prob.T, prob.N, size(prob.k, 2), tol; threads=Threads.nthreads())
  out = Dict{String,Any}()
  for (suffix, ci) in (("", c.trafo), ("_adj", c.adjoint))
    out["rule_W"*suffix] = ci.W; out["rule_sigma"*suffix] = ci.sigma
    out["rule_beta"*suffix] = ci.β; out["rule_grid"*suffix] = ci.grid
  end
  out["rule_window"] = c.trafo.window
  out["rule_tol"] = c.trafo.tol
  merge!(out, filter(kv -> startswith(kv[1], "rep_"), rep))
  dev = String[]
  for f in ("W", "W_adj", "grid", "grid_adj", "sigma", "sigma_adj", "beta", "beta_adj")
    haskey(out, "rep_" * f) || continue
    a, b = out["rep_"*f], out["rule_"*f]
    same = f in ("sigma", "sigma_adj", "beta", "beta_adj") ? isapprox(a, b; rtol=1e-2) : a == b
    same || push!(dev, "$f $(a isa Vector ? join(a, "x") : a) ≠ $(b isa Vector ? join(b, "x") : b)")
  end
  out["rule_dev"] = join(dev, "; ")
  # NFFT.jl without the option gridsize, with a rule whose grid is not from {2, 3, 5} (ducc: 7, 11),
  # builds the next larger grid
  out["grid_rounded"] = task["lib"] == "NFFT" && haskey(out, "rep_grid") &&
                        (out["rep_grid"] != out["rule_grid"] || out["rep_grid_adj"] != out["rule_grid_adj"])
  out
end

function main()
  results = Dict{String,Any}[]
  cache = Dict{Any,Problem}()
  info = libinfo()
  for task in CFG["task"]
    r = Dict{String,Any}(task)
    r["threads"] = Threads.nthreads()
    merge!(r, Dict("lib_" * k => replace(v, homedir() => "~") for (k, v) in info))
    reason = supports(task)
    if reason !== nothing
      r["skipped"] = reason
    else
      try
        # one problem at a time: tasks come grouped by problem, and large ones take gigabytes
        haskey(cache, problem_key(task)) || empty!(cache)
        prob = get!(() -> Problem(task), cache, problem_key(task))
        merge!(r, measure(prob, task))
      catch e
        e isa Unsupported ? (r["skipped"] = e.reason) : (r["error"] = first(sprint(showerror, e), 300))
      end
    end
    push!(results, r)
    open(io -> TOML.print(io, Dict("result" => results)), ARGS[2], "w")
    progress(r)
  end
end

function progress(r)
  lib = haskey(r, "rule") && r["lib"] == "NFFT" ? "NFFT[" * r["rule"] * "]" : r["lib"]
  haskey(r, "fft_flag") && r["lib"] != "NFFT" && (lib *= "/" * lowercase(r["fft_flag"][1:4]))
  head = @sprintf("  %-14s %-5s D=%d N=%-8d J=%-7d L=%-4d %-9s %-7s t=%-2d tol=%.0e",
                  lib, r["transform"], length(r["N"]), r["N"][1], r["J"], get(r, "L", 1),
                  r["nodes"], r["T"], r["threads"], r["tol"])
  if haskey(r, "skipped")
    println(head, "  skipped: ", r["skipped"])
  elseif haskey(r, "error")
    println(head, "  ERROR: ", r["error"])
  else
    @printf("%s  err %.1e/%.1e  plan %8.2f  trafo %8.2f  adj %8.2f ms\n", head, r["err_trafo"],
            r["err_adj"], 1e3r["t_plan"], 1e3r["t_trafo"], 1e3r["t_adj"])
  end
  flush(stdout)
end

main()
