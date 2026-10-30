# Parameter rules of other NUFFT libraries: window, width W and grid (oversampling factor σ) that
# library X would choose for a problem, so that NFFT.jl can run with X's parameters (suite
# `decomposition`). Standard library only; included by the worker and by the orchestrator.
#
#   Rules.choose(rule, T, N, J, tol; threads) -> (trafo = c, adjoint = c)
#
# with c = (; window, β, e0, W, sigma, tol, grid): `sigma` the nominal oversampling factor of the rule,
# `tol` the tolerance X works with (raised or clamped as X or its adapter does), `grid` the
# oversampled grid per dimension as X builds it. The trafo is X's type 2, the adjoint its
# type 1. Windows (z in grid units, |z| < W/2, x = 2z/W):
#   "es"       exp(β((1 − x²)^e0 − 1))                     FINUFFT ≤ 2.4, cuFINUFFT, ducc, mlx-nufft
#   "pswf"     ψ₀ᶜ(x) with c = β, the prolate spheroidal wave function of order 0     FINUFFT ≥ 2.5
#   "kb-sinh"  sinh(β√(1 − x²))/√(1 − x²), truncated at |x| = 1                      NFFT3, NonuniformFFTs
module Rules

using LinearAlgebra: SymTridiagonal, eigen

const RULES = ["FINUFFT 2.5", "FINUFFT 2.6-dev", "cuFINUFFT 2.5", "ducc 0.42", "NonuniformFFTs 0.9", "NFFT3",
               "mlx-nufft 0.3"]
# the library of the benchmark whose own choice a rule describes
const RULE_OF_LIB = Dict("FINUFFT 2.5" => "FINUFFT 2.5", "FINUFFT" => "FINUFFT 2.6-dev", "ducc" => "ducc 0.42",
                         "NonuniformFFTs" => "NonuniformFFTs 0.9", "mlx-nufft" => "mlx-nufft 0.3")

choice(window, β, e0, W, σ, grid, tol) = (; window, β=Float64(β), e0=Float64(e0), W=Int(W), sigma=Float64(σ), tol=Float64(tol),
                                     grid=collect(Int, grid))

function choose(rule, ::Type{T}, N, J, tol; threads=8) where {T}
  f = RULE_FUNCTIONS[rule]
  (trafo=f(T, Tuple(N), J, Float64(tol), 2, threads), adjoint=f(T, Tuple(N), J, Float64(tol), 1, threads))
end

# smallest even integer ≥ n with prime factors 2, 3, 5 (FINUFFT next235even, mlx-nufft next235even)
function next235even(n)
  n <= 2 && return 2
  n += isodd(n)
  while true
    m = n
    for p in (2, 3, 5)
      while m % p == 0
        m ÷= p
      end
    end
    m == 1 && return n
    n += 2
  end
end

# ── FINUFFT 2.5.1 (CPU, FFTW build, more than one thread) ─────────────────────────────────────────
# σ: include/finufft/heuristics.hpp, bestUpsamplingFactor → bestUpsamplingFactorMultithread, chosen
# in setpts from the density J/N (src/finufft_core.cpp). W = ns: src/common/kernel.cpp
# theoretical_kernel_ns (kerformula 8, the default), clipped in src/finufft_core.cpp
# setup_spreadinterp to [2, 16] and for Float32 at σ < 1.4 to ≤ 8. β: set_kernel_shape_given_ns,
# kerformula 8, β = πW(1 − 1/(2σ)) − 0.05. Grid: set_nf_type12, next235even(max(⌈σN⌉, 2W)). The
# tolerance is the plan's, in the type of the plan.
function finufft251_sigma(::Type{T}, D, type, density, tol, threads) where {T}
  (tol <= 1e-9 || tol <= 100 * eps(T)) && return 2.0
  threads == 1 && error("the single-thread table of FINUFFT 2.5.1 is not ported")
  if T == Float32
    D == 3 && type == 1 && tol <= 1e-6 && density >= 32 && return 2.0
    return 1.25
  end
  D == 3 || return 1.25
  steps = type == 1 ? ((16, 1e-5), (8, 1e-7), (2, 1e-8), (1, 1e-9)) :
                      ((16, 1e-5), (8, 1e-6), (4, 1e-7), (2, 1e-8), (1, 1e-9))
  any(((d, e),) -> density >= d && tol <= e, steps) ? 2.0 : 1.25
end

finufft_tolfac(D, type) = 0.18 * 1.4^(D - 1) * (type == 3 ? 1.4 : 1.0)
theoretical_ns(tol, D, type, σ) = ceil(Int, log(finufft_tolfac(D, type) / tol) / (π * sqrt(1 - 1 / σ)) + 1)
finufft_beta(ns, σ) = π * ns * (1 - 1 / (2σ)) - 0.05

function finufft251(::Type{T}, N, J, tol, type, threads) where {T}
  tol = Float64(T(tol))
  σ = finufft251_sigma(T, length(N), type, J / prod(N), tol, threads)
  tol = max(tol, Float64(eps(T)))
  ns = clamp(theoretical_ns(tol, length(N), type, σ), 2, 16)
  T == Float32 && σ < 1.4 && (ns = min(ns, 8))
  choice("pswf", finufft_beta(ns, σ), 0, ns, σ, [next235even(max(ceil(Int, σ * n), 2ns)) for n in N], tol)
end

# ── FINUFFT 2.6.0-dev (main at da76a57, the build of the benchmark) ───────────────────────────────
# σ: include/finufft/heuristics.hpp best_type12, a cost model minimised over the widths W from the
# smallest feasible σ (analytic_upsampfac) down to the width at σ = 2.5; feasibility, width clamp,
# rounding floor and grid: include/finufft_common/kernel.h and src/common/kernel.cpp
# (upsampfac_feasible, clamp_kernel_ns, float_max_ns_cc, lowest_sigma, fine_grid_len), constants
# include/finufft_common/constants.h. SIMD widths of include/finufft/simd.hpp on NEON (128 bit: 2
# Float64, 4 Float32). Kernel and β as in 2.5.1 (kerformula 8). Clang contracts a·b + c into an FMA
# on ARM (-ffp-contract=on); where a choice sits on a rounding boundary (σ of smallest_sigma_for_ns,
# where the width jumps), the rule does the same.
const MIN_AUTO_UPSAMPFAC = 1.25
const MAX_AUTO_UPSAMPFAC = 2.5
const MAX_CHECK_SIGMA = 2.0

max_nspread(T) = T == Float32 ? 12 : 16
float_max_ns_cc(σ, D) = 8 + floor(Int, max(0.0, (σ - 0.975 - 0.125D) * (1.0 / 0.075)))
function clamp_ns(ns, σ, D, ::Type{T}) where {T}
  ns = clamp(ns, 2, max_nspread(T))
  T == Float32 ? min(ns, float_max_ns_cc(σ, D)) : ns
end
fine_grid_len(σ, n, ns) = next235even(max(ceil(Int, σ * n), 2ns))

function smallest_sigma_for_ns(tol, D, type, ns)
  tolfac = finufft_tolfac(D, type)
  tol <= 0 && return MAX_AUTO_UPSAMPFAC
  tol >= tolfac && return 1.01
  u = log(tolfac / tol) / ((ns - 1.0) * π)
  u >= 1 && return MAX_AUTO_UPSAMPFAC
  min(1 / fma(-u, u, 1.0), MAX_AUTO_UPSAMPFAC)
end

function lowest_sigma(tol, D, ns, epsm, gridlen)
  r = tol / (0.48 * epsm * gridlen)
  r <= 0.5 && return Inf
  pure = smallest_sigma_for_ns(tol, D, 1, ns)
  r >= 10 && return min(pure, MAX_CHECK_SIGMA)
  a2, a1, a0 = ns > 8 ? (0.014, 0.291, -0.043) : (0.555, -0.290, 0.071)
  corr = fma(fma(a2, 1 / r, a1), 1 / r, a0)
  min(pure + max(corr, 0.0), MAX_CHECK_SIGMA)
end

function upsampfac_feasible(σ, tol, D, type, ::Type{T}, maxN) where {T}
  nst = theoretical_ns(tol, D, type, σ)
  ns = clamp_ns(nst, σ, D, T)
  ns < nst && return false
  nf = fine_grid_len(σ, maxN, ns)
  min(lowest_sigma(tol, D, ns, Float64(eps(T)), nf), MAX_CHECK_SIGMA) <= σ
end

function analytic_upsampfac(tol, D, type, ::Type{T}, maxN) where {T}
  feasible(σ) = upsampfac_feasible(σ, tol, D, type, T, maxN)
  feasible(MIN_AUTO_UPSAMPFAC) && return MIN_AUTO_UPSAMPFAC
  feasible(MAX_AUTO_UPSAMPFAC) || return MAX_AUTO_UPSAMPFAC
  lo, hi = MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC
  for _ in 1:40
    mid = (lo + hi) / 2
    feasible(mid) ? (hi = mid) : (lo = mid)
  end
  hi
end

function spread_row(ns, ::Type{T}) where {T}
  w = T == Float32 ? 4 : 2
  max(cld(2ns, w), 1)
end

function finufft26_sigma(::Type{T}, N, J, tol, type, threads) where {T}
  D = length(N)
  maxN = maximum(N)
  width(σ) = clamp_ns(theoretical_ns(tol, D, type, σ), σ, D, T)
  c = 2.0 * sqrt(max(1, threads))
  function cost(σ, ns)
    G = prod(Float64(fine_grid_len(σ, n, ns)) for n in N)
    J * spread_row(ns, T) * Float64(ns)^(D - 1) + c * G * log2(G)
  end
  σmin = analytic_upsampfac(tol, D, type, T, maxN)
  nsmin = width(σmin)
  best, bestcost = σmin, cost(σmin, nsmin)
  upsampfac_feasible(σmin, tol, D, type, T, maxN) || return best
  for nst in nsmin-1:-1:width(MAX_AUTO_UPSAMPFAC)
    s = clamp(smallest_sigma_for_ns(tol, D, type, nst), σmin, MAX_AUTO_UPSAMPFAC)
    upsampfac_feasible(s, tol, D, type, T, maxN) || continue
    cs = cost(s, width(s))
    cs < bestcost && ((best, bestcost) = (s, cs))
  end
  best
end

# FINUFFT 2.6 rejects a tol below its rounding floor (include/finufft/setpts.hpp check_sigma); the
# benchmark adapter then doubles tol until the plan is accepted, and so does the rule.
function finufft26(::Type{T}, N, J, tol, type, threads) where {T}
  D = length(N)
  tol = Float64(T(tol))
  while true
    tol < eps(T) && error("FINUFFT 2.6 rejects tol below eps of the type")
    σ = finufft26_sigma(T, N, J, tol, type, threads)
    ns = clamp_ns(theoretical_ns(tol, D, type, σ), σ, D, T)
    grid = [fine_grid_len(σ, n, ns) for n in N]
    if lowest_sigma(tol, D, ns, Float64(eps(T)), maximum(grid)) <= σ || tol >= 0.25
      return choice("pswf", finufft_beta(ns, σ), 0, ns, σ, grid, tol)
    end
    tol = Float64(T(2tol))
  end
end

# ── cuFINUFFT 2.5.1 ───────────────────────────────────────────────────────────────────────────────
# σ = 2 for types 1 and 2 (include/cufinufft/impl.h). W and β: src/cuda/spreadinterp.cpp
# setup_spreader, in the precision of the plan: W = ⌈−log10(tol/10)⌉ clipped to [2, 16], ES window
# with β = 2.30W (W = 2, 3, 4: 2.20, 2.26, 2.38). Grid: src/cuda/common.cu set_nf_type12,
# next235beven(⌈2N⌉, 1) = next235even.
function cufinufft251(::Type{T}, N, J, tol, type, threads) where {T}
  e = max(T(tol), eps(T))
  ns = clamp(ceil(Int, -log10(e / T(10))), 2, 16)
  βn = ns == 2 ? 2.20 : ns == 3 ? 2.26 : ns == 4 ? 2.38 : 2.30
  choice("es", Float64(T(βn)) * ns, 0.5, ns, 2.0, [next235even(max(ceil(Int, 2.0 * n), 2ns)) for n in N], e)
end

# ── ducc 0.42.0 (branch ducc0 at 31d9fe9, the build of the benchmark, through Ducc0.jl) ──────────────
# Kernels: KernelDB of src/ducc0/math/gridding_kernel.cc (ducc_kernels_0.42.csv), per W ≤ 16 (Float32:
# 8) the smallest σ in [1.1, 2.6] whose ε reaches tol (gridding_kernel.h getAvailableKernels).
# Choice among them: src/ducc0/nufft/nufft_common.h findNufftParameters, a cost model in the number of
# threads; vector length of NEON (2 Float64, 4 Float32). Grid 2·good_size_complex(⌊Nσ/2⌋ + 1) ≥ 16,
# sizes with factors 2, 3, 5, 7, 11 (src/ducc0/fft/fft.h). Window esknew(v, βW, e0). One plan
# serves both directions; Float64 accumulates in Float64, so the cost is the same for both. The
# benchmark adapter raises tol to 1.1 × the best ε of the table.
const DUCC_DB = let
  rows = filter(l -> !startswith(l, "#"), readlines(joinpath(@__DIR__, "ducc_kernels_0.42.csv")))[2:end]
  map(rows) do l
    c = split(l, ",")
    (W=parse(Int, c[1]), ofactor=parse(Float64, c[2]), epsilon=parse(Float64, c[3]), beta=parse(Float64, c[4]),
     e0=parse(Float64, c[5]), ndim=parse(Int, c[6]), singleprec=c[7] == "true")
  end
end

function good_size_complex(n)
  n <= 12 && return n
  best = 2n
  f11 = 1
  while f11 < best
    f117 = f11
    while f117 < best
      f1175 = f117
      while f1175 < best
        x = f1175
        while x < n
          x *= 2
        end
        while true
          if x < n
            x *= 3
          elseif x > n
            x < best && (best = x)
            isodd(x) && break
            x >>= 1
          else
            return n
          end
        end
        f1175 *= 5
      end
      f117 *= 7
    end
    f11 *= 11
  end
  best
end

ducc_best_epsilon(D, sp) = minimum(k.epsilon for k in DUCC_DB if k.ndim == D && k.singleprec == sp &&
                                   1.1 <= k.ofactor <= 2.6)

function ducc042(::Type{T}, N, J, tol, type, threads) where {T}
  D = length(N); sp = T == Float32
  tol = max(tol, 1.1 * ducc_best_epsilon(D, sp))
  ofc = fill(2.6, 20); idx = zeros(Int, 20)
  for (i, k) in enumerate(DUCC_DB)
    if k.ndim == D && k.singleprec == sp && k.W <= (sp ? 8 : 16) && k.epsilon <= tol &&
       k.ofactor <= ofc[k.W+1] && k.ofactor >= 1.1
      ofc[k.W+1] = k.ofactor; idx[k.W+1] = i
    end
  end
  vlen = sp ? 4 : 2
  sigmoid(x, m, s) = 1 + (x - 1) / (1 + ((x - 1) / (m - 1))^s)^(1 / s)
  best = nothing; mincost = Inf
  for i in filter(!=(0), idx)
    k = DUCC_DB[i]
    nvec = cld(k.W, vlen)
    grid = [max(2 * good_size_complex(floor(Int, n * k.ofactor * 0.5) + 1), 16) for n in N]
    G = prod(Float64, grid)
    fftcost = G / 2048.0^2 * (log(G) / log(2048.0^2)) * 0.0693 / sigmoid(threads, 6, 2)
    kernelpoints = nvec * vlen * k.W^(D - 1)
    gridcost = 2.2e-10 * J * (kernelpoints + D * nvec * (k.W + 3) * vlen) / threads
    if fftcost + gridcost < mincost
      mincost = fftcost + gridcost
      best = choice("es", k.beta * k.W, k.e0, k.W, k.ofactor, grid, tol)
    end
  end
  best
end

# ── NonuniformFFTs.jl 0.9.8 ───────────────────────────────────────────────────────────────────────
# No tolerance parameter: the benchmark adapter (worker/adapters/nonuniformffts.jl) takes the half
# support M = ⌊(⌈log10(1/tol)⌉ + 1)/2⌋ (NFFT.jl 0.14), W = 2M, with the default σ = 2 and the
# default kernel BackwardsKaiserBesselKernel (src/NonuniformFFTs.jl default_kernel). Grid
# nextprod((2, 3, 5), ⌊σN⌋) (src/plan.jl); β = πaγ, a = M(2 − 1/σ), γ = max(0.995, √(1 − 0.3/a²)),
# σ the actual Ñ/N (src/Kernels/kaiser_bessel_backwards.jl optimal_kernel).
function nonuniformffts098(::Type{T}, N, J, tol, type, threads) where {T}
  M = (ceil(Int, log10(1 / tol)) + 1) ÷ 2
  grid = [nextprod((2, 3, 5), floor(Int, 2.0 * n)) for n in N]
  σ = maximum(grid ./ collect(N))
  a = M * (2 - 1 / σ)
  β = π * a * max(0.995, sqrt(1 - 0.3 / a^2))
  choice("kb-sinh", Float64(T(β)), 0, 2M, 2.0, grid, tol)
end

# ── NFFT3 (github.com/NFFT/nfft, master at 6506edc) ──────────────────────────────────────────────
# nfft_init (kernel/nfft/nfft.c): n = 2·nextpow2(N), m = WINDOW_HELP_ESTIMATE_m of the default
# Kaiser–Bessel window (include/infft.h: 8 for Float64, 4 for Float32), independent of any
# tolerance. The window reaches WINDOW_STENCIL_REACH = m + 1 grid points to each side, W = 2m + 2,
# sinh(b√((m+1)² − (nx)²))/√(…) with b = π(2 − 1/σ) (kb_phi), i.e. β = b(m + 1).
function nfft3(::Type{T}, N, J, tol, type, threads) where {T}
  m = T == Float32 ? 4 : 8
  grid = [2 * nextpow(2, n) for n in N]
  σ = grid[1] / N[1]
  choice("kb-sinh", π * (2 - 1 / σ) * (m + 1), 0, 2m + 2, σ, grid, tol)
end

# ── mlx-nufft 0.3.3 (github.com/martinlachaine/mlx-nufft, edce21c) ───────────────────────────────
# tol raised to 1e-6 (mlx_nufft/api.py _EPS_FLOOR). σ: mlx_nufft/nd.py _resolve_upsampfac (2 below
# 3D or 2^15 modes; 1.25 by type, tol and density ρ = J/(2^D N)). W and β: mlx_nufft/sizing.py
# kernel_params, the rule of FINUFFT ≤ 2.4 / cuFINUFFT with W capped at σ < 2 to 8 (3D) or 10. Grid
# next235even(max(2W, ⌈σN⌉)).
function mlx_sigma(D, type, tol, N, J)
  D < 3 && return 2.0
  prod(N) < 2^15 && return 2.0
  ρ = J / (2.0^D * prod(N))
  if type == 2
    tol >= 1e-3 && return 1.25
    tol >= 1e-4 && ρ < 0.3 && return 1.25
    return 2.0
  end
  ρ >= 0.1 && return 2.0
  tol >= 1e-3 && return 1.25
  tol >= 1e-4 && prod(N) >= 2^24 && return 1.25
  2.0
end

function mlxnufft033(::Type{T}, N, J, tol, type, threads) where {T}
  tol = max(tol, 1e-6)
  D = length(N)
  σ = mlx_sigma(D, type, tol, N, J)
  ns = σ == 2 ? ceil(Int, -log10(tol / 10)) : ceil(Int, -log(tol) / (π * sqrt(1 - 1 / σ)))
  ns = clamp(ns, 2, 16)
  σ < 2 && (ns = min(ns, D >= 3 ? 8 : 10))
  βn = ns == 2 ? 2.20 : ns == 3 ? 2.26 : ns == 4 ? 2.38 : 2.30
  σ != 2 && (βn = 0.97 * π * (1 - 1 / (2σ)))
  choice("es", βn * ns, 0.5, ns, σ, [next235even(max(2ns, ceil(Int, σ * n))) for n in N], tol)
end

const RULE_FUNCTIONS = Dict("FINUFFT 2.5" => finufft251, "FINUFFT 2.6-dev" => finufft26,
                            "cuFINUFFT 2.5" => cufinufft251, "ducc 0.42" => ducc042,
                            "NonuniformFFTs 0.9" => nonuniformffts098, "NFFT3" => nfft3,
                            "mlx-nufft 0.3" => mlxnufft033)

# ── windows ───────────────────────────────────────────────────────────────────────────────────────

# ψ₀ᶜ on [−1, 1], normalised to ψ(0) = 1: expansion in normalised Legendre polynomials of even degree,
# the eigenvector of the smallest eigenvalue of the prolate differential operator
# −((1 − x²)ψ′)′ + c²x²ψ (Xiao, Rokhlin & Yarvin 2001).
function pswf0(c)
  n = ceil(Int, c) + 40                                  # even degrees 0, 2, …, 2(n−1)
  k = 0:2:2(n-1)
  d = [k * (k + 1) + c^2 * (2k^2 + 2k - 1) / ((2k + 3) * (2k - 1)) for k in k]
  e = [c^2 * (k + 1) * (k + 2) / ((2k + 3) * sqrt((2k + 1) * (2k + 5))) for k in k[1:end-1]]
  v = eigen(SymTridiagonal(d, e)).vectors[:, 1]
  coef = v .* sqrt.(k .+ 0.5)                            # in the unnormalised P_k
  function legendre_even(x)
    s = 0.0; p0 = 1.0; p1 = x
    s += coef[1] * p0
    for j in 1:2(n-1)
      p0, p1 = p1, ((2j + 1) * x * p1 - j * p0) / (j + 1)
      iseven(j + 1) && (s += coef[(j+1)÷2+1] * p1)
    end
    s
  end
  ψ0 = legendre_even(0.0)
  x -> abs(x) >= 1 ? 0.0 : legendre_even(x) / ψ0
end

"""
    window(c) -> φ(z, W)

The window of choice `c` for `NFFT.CustomWindow`.
"""
function window(c)
  β, e0 = c.β, c.e0
  if c.window == "es"
    return (z, W) -> (t = 1 - (2z / W)^2; t > 0 ? exp(β * (t^e0 - 1)) : zero(z))
  elseif c.window == "kb-sinh"
    edge = sinh(β)
    return (z, W) -> (t = 1 - (2z / W)^2; t > 0 ? (s = sqrt(t); (s < 1e-8 ? β : sinh(β * s) / s) / edge) : zero(z))
  elseif c.window == "pswf"
    ψ = pswf0(β)
    return (z, W) -> ψ(2z / W)
  end
  error("unknown window $(c.window)")
end

end
