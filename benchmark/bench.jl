# Benchmark suite of NFFT.jl (see README.md).
#
#   julia benchmark/bench.jl list
#   julia benchmark/bench.jl run <suite> [options]       run a suite, CSV to benchmark/out/
#   julia benchmark/bench.jl ab <rev> [suite] [options]  working tree against a git revision (default suite: fast)
#   julia benchmark/bench.jl compare <new.csv> <old.csv> time ratios of two runs
#   julia benchmark/bench.jl report [machine]            figures and data for the documentation
#
# Options: --libs NFFT,FINUFFT  --dims 1,2  --tols 1e-4,1e-8  --T Float32  --threads 1,8
#          --budget 0.1  --rounds 2 (ab)  --save (results/<machine>/<suite>.csv, for the docs)

include(joinpath(@__DIR__, "src", "Benchmarks.jl"))
using .Benchmarks: Benchmarks, SUITES, SUITE_HELP, LIBS, run_tasks, write_csv, read_csv, compare,
                   ab, meta_info, machine, ROOT, REPO

function parse_opts(args)
  o = Dict{Symbol,Any}(); pos = String[]
  i = 1
  while i <= length(args)
    a = args[i]
    if a == "--save"
      o[:save] = true; i += 1; continue
    elseif startswith(a, "--")
      v = split(args[i+1], ","); k = Symbol(a[3:end])
      o[k] = k in (:dims, :threads, :rounds) ? parse.(Int, v) : k in (:tols, :budget) ? parse.(Float64, v) :
             String.(v)
      k == :budget && (o[k] = o[k][1])
      i += 2
    else
      push!(pos, a); i += 1
    end
  end
  o, pos
end

function save(results, suite, args, o)
  meta = meta_info(suite, args)
  stamp = Dates.format(Benchmarks.now(), "yyyy-mm-dd_HHMMSS")
  file = write_csv(joinpath(ROOT, "out", "$stamp-$suite.csv"), results; meta)
  println("results: ", relpath(file, pwd()))
  if get(o, :save, false)
    for s in unique(first(split(r["suite"], ":")) for r in results)
      rs = filter(r -> first(split(r["suite"], ":")) == s, results)
      file = joinpath(ROOT, "results", machine(), "$s.csv")
      # measurements of an earlier run that this run does not repeat are kept
      if isfile(file)
        new = Set(Benchmarks.key.(rs))
        _, old = read_csv(file)
        rs = [filter(r -> Benchmarks.key(r) ∉ new, old); rs]
      end
      f = write_csv(file, rs; meta=merge(meta, Dict("suite" => s)))
      println("saved:   ", relpath(f, pwd()))
    end
  end
end

using Dates
function main(args)
  Benchmarks.LOAD_START[] = Benchmarks.loadavg()
  o, pos = parse_opts(args)
  cmd = isempty(pos) ? "list" : pos[1]
  if cmd == "list"
    println("suites:")
    for s in sort(collect(keys(SUITES)))
      println("  ", rpad(s, 9), SUITE_HELP[s])
    end
    println("libraries: ", join(sort(collect(keys(LIBS))), ", "), ", NFFT@<git revision>")
  elseif cmd == "run"
    suite = get(pos, 2, "quick")
    t = @elapsed res = run_tasks(SUITES[suite](o))
    save(res, suite, args, o)
    println("time: ", round(t, digits=1), " s")
  elseif cmd == "ab"
    rev = pos[2]; suite = get(pos, 3, "fast")
    o[:libs] = ["NFFT"]
    t = @elapsed (cur, old) = ab(rev, SUITES[suite](o); rounds=get(o, :rounds, [2])[1])
    save(cur, "ab-now", args, Dict()); save(old, "ab-$rev", args, Dict())
    println("time: ", round(t, digits=1), " s")
  elseif cmd == "compare"
    _, new = read_csv(pos[2]); _, old = read_csv(pos[3])
    compare(new, old; labels=("new", "old"))
  elseif cmd == "report"
    env = joinpath(ROOT, "envs", "report")
    Benchmarks.ensure_env(env)
    run(`$(Benchmarks.julia()) --project=$env $(joinpath(ROOT, "src", "report.jl")) $(get(pos, 2, machine()))`)
  else
    error("unknown command $cmd")
  end
end

main(ARGS)
