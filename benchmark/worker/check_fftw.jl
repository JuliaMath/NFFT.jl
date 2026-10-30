# Check that the worker's FFTW handling (fftw.jl) leaves FFTW's threaded plans as fast as in a
# fresh process: the same threaded plan is timed once in a fresh process and once in a process
# that first calls `forget_fftw_wisdom`, as the worker does before its first plan.
#
#   julia --project=benchmark/envs/finufft -t 8 benchmark/worker/check_fftw.jl
#
# FFTW.jl's FFTW (plans compared as printed by FFTW, and timed) and FINUFFT's own FFTW (timed,
# through a type-1 transform). Exits with an error if a plan differs or a time ratio exceeds TOL.
# Run with several threads: at 1 thread FFTW plans no threaded solvers and the check is void.

using FFTW
import FINUFFT
include(joinpath(@__DIR__, "fftw.jl"))   # defines functions only, calls no FFTW

const TOL = 1.25
const CASES = [("fftw", ComplexF64, (2^17,)), ("fftw", ComplexF32, (2^17,)), ("fftw", ComplexF64, (1024, 1024)),
               ("finufft", Float64, (2^16,)), ("finufft", Float32, (512, 512))]

function child(mode, lib, T, n)
  mode == "worker" && forget_fftw_wisdom()
  if lib == "fftw"
    FFTW.set_num_threads(Threads.nthreads())
    x = zeros(T, n)
    p = plan_fft!(x; flags=FFTW.ESTIMATE)
    f = () -> p * x
    desc = replace(string(p), r"\n\s*" => " ")
  else
    M = 2 * prod(n)
    k = [T(2π) .* (rand(T, M) .- T(0.5)) for _ in n]; c = rand(Complex{T}, M)
    tol = T == Float64 ? 1e-8 : 1e-4   # Float32: FINUFFT rejects tolerances below its rounding floor
    f = length(n) == 1 ? () -> FINUFFT.nufft1d1(k[1], c, 1, tol, n[1]; nthreads=Threads.nthreads()) :
                         () -> FINUFFT.nufft2d1(k[1], k[2], c, 1, tol, n...; nthreads=Threads.nthreads())
    desc = "-"
  end
  f()
  ts = Float64[]; t0 = time()
  while length(ts) < 5 || time() - t0 < 0.5
    push!(ts, @elapsed f())
  end
  t = minimum(ts)
  println(t, "\t", desc)
end

function main()
  if length(ARGS) == 4
    return child(ARGS[1], ARGS[2], eval(Meta.parse(ARGS[3])), Tuple(parse.(Int, split(ARGS[4], "x"))))
  end
  Threads.nthreads() > 1 || @warn "1 thread: FFTW plans no threaded solvers, the check is void"
  julia = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t $(Threads.nthreads())`
  ok = true
  for (lib, T, n) in CASES
    r = Dict(mode => split(readchomp(`$julia $(@__FILE__) $mode $lib $T $(join(n, "x"))`), "\t"; limit=2)
             for mode in ("fresh", "worker"))
    tf, tw = parse(Float64, r["fresh"][1]), parse(Float64, r["worker"][1])
    same = r["fresh"][2] == r["worker"][2]
    good = same && tw / tf <= TOL
    ok &= good
    println(rpad("$lib $T $(join(n, "×"))", 28), "fresh ", round(1e3tf, digits=3), " ms  worker ",
            round(1e3tw, digits=3), " ms  ratio ", round(tw / tf, digits=2), same ? "" : "  PLANS DIFFER",
            good ? "" : "  FAIL")
  end
  ok || error("the worker's FFTW state slows FFTW down (threads: $(Threads.nthreads()))")
  println("ok (threads: $(Threads.nthreads()))")
end

main()
