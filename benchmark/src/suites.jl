# Benchmark suites: each returns a list of tasks (Dicts), one per library and measurement. The
# options given on the command line (`libs`, `dims`, `tols`, `T`, `threads`, `budget`) replace the
# defaults of a suite where they apply.

const SIZES = Dict(1 => [2^18], 2 => [512, 512], 3 => [64, 64, 64])
const COMPETITORS = ["NFFT", "FINUFFT", "ducc"]

task(; lib, N, J=prod(N), transform="nfft", L=1, nodes="uniform", T="Float64", tol=1e-8,
     threads=8, budget=0.1, seed=1234, suite="", stages=false, plans=3) =
  Dict{String,Any}("lib" => lib, "transform" => transform, "N" => collect(N), "J" => J, "L" => L,
                   "nodes" => nodes, "T" => T, "tol" => tol, "threads" => threads, "budget" => budget,
                   "seed" => seed, "suite" => suite, "stages" => stages, "plans" => plans)

opt(o, key, default) = get(o, key, default)
# Performance cores only: on Apple silicon the efficiency cores slow down threaded loops with static
# work distribution.
function cores()
  Sys.isapple() || return Sys.CPU_THREADS
  something(tryparse(Int, try readchomp(`sysctl -n hw.perflevel0.physicalcpu`) catch; "" end), Sys.CPU_THREADS)
end
default_threads() = min(8, cores())
tolsweep(T) = T == "Float32" ? [10.0^-e for e in 2:6] : [10.0^-e for e in 2:14]

const SUITES = Dict{String,Any}()
const SUITE_HELP = Dict{String,String}()
macro suite(name, help, body)
  quote
    SUITE_HELP[$name] = $help
    SUITES[$name] = $(esc(body))
  end
end

@suite "fast" "agent check of NFFT.jl alone: D = 1…3, tol 1e-4/1e-8, one plan (≈ 15 s)" o -> [
  task(; lib, N=SIZES[D], tol, T, threads=opt(o, :threads, [default_threads()])[1], budget=opt(o, :budget, 0.1),
       plans=1, suite="fast")
  for D in opt(o, :dims, [1, 2, 3]) for T in opt(o, :T, ["Float64"])
  for tol in opt(o, :tols, [1e-4, 1e-8]) for lib in opt(o, :libs, ["NFFT"])]

@suite "quick" "agent check: NFFT vs FINUFFT vs ducc, D = 1…3, tol 1e-4/1e-8/1e-12 (≈ 1 min)" o -> [
  task(; lib, N=SIZES[D], tol, T, threads=opt(o, :threads, [default_threads()])[1], budget=opt(o, :budget, 0.1), suite="quick")
  for D in opt(o, :dims, [1, 2, 3]) for T in opt(o, :T, ["Float64"])
  for tol in opt(o, :tols, [1e-4, 1e-8, 1e-12]) for lib in opt(o, :libs, COMPETITORS)]

@suite "pareto" "time against measured error, tolerance sweep, Float64 and Float32, D = 1…3" o -> [
  task(; lib, N=SIZES[D], tol, T, threads, budget=opt(o, :budget, 0.1), suite="pareto")
  for threads in opt(o, :threads, [default_threads()]) for D in opt(o, :dims, [1, 2, 3])
  for T in opt(o, :T, ["Float64", "Float32"]) for tol in opt(o, :tols, tolsweep(T))
  for lib in opt(o, :libs, [COMPETITORS; "NonuniformFFTs"])]

@suite "threads" "strong scaling over the number of threads, tol 1e-4 and 1e-8" o -> [
  task(; lib, N=SIZES[D], tol, threads, budget=opt(o, :budget, 0.1), suite="threads")
  for threads in opt(o, :threads, unique(min.([1, 2, 4, 8, 16], cores())))
  for D in opt(o, :dims, [1, 2, 3]) for tol in opt(o, :tols, [1e-4, 1e-8])
  for lib in opt(o, :libs, COMPETITORS)]

const SWEEP_SIZES = Dict(1 => [[2^e] for e in 10:2:22], 2 => [[n, n] for n in (32, 64, 128, 256, 512, 1024, 2048)],
                         3 => [[n, n, n] for n in (16, 24, 32, 48, 64, 96, 128)])
@suite "sizes" "problem size sweep with J = |N| nodes, tol 1e-8" o -> [
  task(; lib, N, tol, threads=opt(o, :threads, [default_threads()])[1], budget=opt(o, :budget, 0.1), suite="sizes")
  for D in opt(o, :dims, [1, 2, 3]) for N in SWEEP_SIZES[D] for tol in opt(o, :tols, [1e-8])
  for lib in opt(o, :libs, COMPETITORS)]

@suite "nodes" "node distributions (uniform, radial, clustered), D = 2, 3" o -> [
  task(; lib, N=SIZES[D], nodes, tol, threads=opt(o, :threads, [default_threads()])[1], budget=opt(o, :budget, 0.1), suite="nodes")
  for D in opt(o, :dims, [2, 3]) for nodes in ["uniform", "radial", "clustered"]
  for tol in opt(o, :tols, [1e-4, 1e-8]) for lib in opt(o, :libs, COMPETITORS)]

# OCT: 1D A-scans; MRI: multi-coil 2D; 3D volumes
const BATCH_CASES = [("oct", [2048], 2048, [1, 8, 64, 512]), ("mri", [256, 256], 65536, [1, 4, 16, 32]),
                     ("volume", [64, 64, 64], 2^18, [1, 4, 8])]
@suite "batch" "many transforms with the same nodes (OCT, multi-coil MRI, volumes)" o -> [
  task(; lib, N, J, L, tol, threads=opt(o, :threads, [default_threads()])[1], budget=opt(o, :budget, 0.1),
       suite="batch:" * name)
  for (name, N, J, Ls) in BATCH_CASES if length(N) in opt(o, :dims, [1, 2, 3])
  for L in Ls for tol in opt(o, :tols, [1e-4, 1e-8]) for lib in opt(o, :libs, COMPETITORS)]

@suite "real" "real data: RNFFT, NFCT, NFST against the complex NFFT of the same size" o -> [
  [task(; lib="NFFT", N=SIZES[D], transform, tol, threads=opt(o, :threads, [default_threads()])[1],
        budget=opt(o, :budget, 0.1), suite="real")
   for D in opt(o, :dims, [1, 2, 3]) for tol in opt(o, :tols, [1e-4, 1e-8])
   for transform in ["nfft", "rnfft", "nfct", "nfst"]];
  [task(; lib, N=SIZES[D], tol, threads=opt(o, :threads, [default_threads()])[1], budget=opt(o, :budget, 0.1), suite="real")
   for D in opt(o, :dims, [1, 2, 3]) for tol in opt(o, :tols, [1e-4, 1e-8]) for lib in ["FINUFFT", "ducc"]];
  [task(; lib="NFFT", N=[2048], J=2048, L=512, transform, tol, threads=opt(o, :threads, [default_threads()])[1],
        budget=opt(o, :budget, 0.1), suite="real:oct")
   for tol in opt(o, :tols, [1e-4, 1e-8]) for transform in ["nfft", "rnfft"]]]

@suite "history" "NFFT.jl against NFFT.jl 0.14 (1 and all threads)" o -> [
  task(; lib, N=SIZES[D], tol, threads, budget=opt(o, :budget, 0.1), suite="history")
  for threads in opt(o, :threads, unique([1, default_threads()])) for D in opt(o, :dims, [1, 2, 3])
  for tol in opt(o, :tols, [10.0^-e for e in 2:2:12]) for lib in opt(o, :libs, ["NFFT", "NFFT 0.14"])]

@suite "nnfft" "NNFFT (type 3) against FINUFFT: time against measured error, D = 1…3" o -> [
  task(; lib, transform="nnfft", N=NNFFT_SIZES[D], J=prod(NNFFT_SIZES[D]), tol, T, threads=opt(o, :threads, [default_threads()])[1],
       budget=opt(o, :budget, 0.1), suite="nnfft")
  for D in opt(o, :dims, [1, 2, 3]) for T in opt(o, :T, ["Float64", "Float32"])
  for tol in opt(o, :tols, T == "Float32" ? [1e-2, 1e-4, 1e-6] : [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
  for lib in opt(o, :libs, ["NFFT", "FINUFFT"])]
# sources and targets: prod(N) each, with the space-frequency extent of an NFFT of size N
const NNFFT_SIZES = Dict(1 => [2^16], 2 => [256, 256], 3 => [32, 32, 32])

const HOMEPAGE = ["pareto", "threads", "sizes", "nodes", "batch", "real", "nnfft", "history"]
@suite "all" "all suites of the documentation (" * join(HOMEPAGE, ", ") * ")" o ->
  reduce(vcat, [SUITES[s](o) for s in HOMEPAGE])
