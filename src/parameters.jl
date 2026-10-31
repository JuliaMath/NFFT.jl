# Parameter choice: window, width W, oversampling σ and the window's shape parameters from a
# table of optimised kernels, by predicted error and a fitted cost model.
#
# data/kernel_table.csv: for each window family, W = 4…16 and σ = 1.20:0.05:2.50 the shape
# parameters that minimise the worst-mode aliasing estimate max_ξ √(Σ_{r≠0} φ̂(ξ+r)²)/|φ̂(ξ)| (est_max),
# the RMS of that ratio over the kept modes (est_rms, the error for random nodes) and the bound for
# any nodes, est_wc = max_ξ Σ_{r≠0} |φ̂(ξ+r)|/|φ̂(ξ)| (nodes at one fractional grid offset, as in
# Cartesian sampling, add the aliases of a mode coherently). Computed by the kernel optimiser of the
# refactoring workspace (bench/micro/kernel_opt.jl, kernel_wc.jl).
# The kernel family with a free exponent and the per-(W, σ) optimisation follow ducc (Arras et al.,
# A&A 646, A58, 2021); the table itself is independent of ducc's.

struct KernelEntry
  window::AbstractWindow
  W::Int
  σ::Float64
  β::Float64
  e0::Float64
  est::Float64        # aliasing bound for any nodes (est_wc of the table)
  amp::Float64        # amplification of the resampling correction, φ̂(0)/φ̂(1/(2σ))
end

window_function(e::KernelEntry) = window_function(e.window, e.W, e.β, e.e0)

# The resampling correction multiplies the modes at the edge of the band by up to A = φ̂(0)/φ̂(1/(2σ)),
# and with them the rounding errors of the grid values: the error grows by ≈ 10·eps(T)·A (Float32,
# W = 10, σ = 1.2: A = 140, measured 1.6e-4 for nodes on grid points). Large for small σ and large W.
function amplification(window, W, σ, β, e0)
  φ̂ = window_quadrature(window_function(window, W, β, e0), W)
  φ̂(0.0) / φ̂(1 / (2σ))
end

const KERNEL_TABLE = let lines = readlines(joinpath(@__DIR__, "..", "data", "kernel_table.csv"))[2:end]
  map(split.(lines, ",")) do v
    w = table_window(v[1]); W = parse(Int, v[2]); σ = parse(Float64, v[3])
    β = parse(Float64, v[4]); e0 = parse(Float64, v[5])
    KernelEntry(w, W, σ, β, e0, parse(Float64, v[8]), amplification(w, W, σ, β, e0))
  end
end

# Cost model (ms, direct + adjoint NFFT): t ≈ Σᵢ cᵢ·tᵢ with the terms tᵢ of `cost_terms` for the
# storage and coefficients c per dimension (D ≤ 3), storage and element type: the defaults of
# `default_cost`, or those set by `set_cost_model!` (`calibrate_cost`).
#
# CPU: t ≈ a·J·W^D (taps) + a1·J·W (window evaluation per node and dimension; the same term as the
# taps in 1D) + b·M·log2(M) (FFT of the oversampled grid, M = Ñ^D) + c (fixed costs: tasks, small
# loops), all nonnegative. Fitted by `calibrate_cost` on an M1 Max with 8 threads, Float64, over the
# node densities J/N = 0.01…100, W = 4:2:14, σ = 1.25…2.5, two sizes per dimension, grids from
# {2, 3, 5}: model/measured median 0.97–0.98, 5–95 % quantiles 0.40–1.31. Per tap a = 0.43 ns (1D),
# 0.057 ns (2D), 0.053 ns (3D); b = 0.17–0.21 ns; c = 0.4–0.7 ms.
const COST_COEF = Dict(1 => [4.30e-07, 0.0, 2.14e-07, 0.403],
                       2 => [5.72e-08, 8.09e-07, 1.73e-07, 0.440],
                       3 => [5.30e-08, 1.44e-06, 2.08e-07, 0.719])

cost_terms(::Type{<:Array}, D, W, M, J, T) = (J * W^D, J * W, M * log2(M), 1.0)
default_cost(::Type{<:Array}, T) = COST_COEF

const COST_MODELS = Dict{Tuple{Core.TypeName,DataType},Dict{Int,Vector{Float64}}}()

# MtlArray{Float32, 2} and MtlArray share a model
storage_key(::Type{S}) where {S} = Base.typename(Base.unwrap_unionall(S))

cost_model(::Type{S}, ::Type{T}) where {S,T} = get(() -> default_cost(S, T), COST_MODELS, (storage_key(S), T))

"""
    NFFT.set_cost_model!(storage, eltype, coef)
    NFFT.set_cost_model!(storage, eltype, nothing)

Use the coefficients `coef` (a `Dict` from the dimension 1, 2, 3 to a tuple, as returned by
[`NFFT.calibrate_cost`](@ref)) when plans with this `storage` and `eltype` choose their parameters;
`nothing` restores the built-in model. The setting holds for the Julia session.
"""
function set_cost_model!(S, ::Type{T}, coef) where {T}
  key = (storage_key(S), T)
  coef === nothing ? delete!(COST_MODELS, key) :
    (COST_MODELS[key] = Dict{Int,Vector{Float64}}(d => collect(Float64, c) for (d, c) in coef))
  nothing
end

# The plan options hold the storage as an abstract `Type`: Array goes through methods that infer
# (juliac --trim), other storage through `device_cost`, which the GPU extension defines.
predicted_cost(S, ::Type{T}, W, M, J, D) where {T} =
  S <: Array ? model_cost(Array, T, W, M, J, D) : device_cost(S, T, W, M, J, D)::Float64

device_cost(S, T, W, M, J, D) = throw(ArgumentError("no cost model for storage = $S"))

function model_cost(::Type{S}, ::Type{T}, W, M, J, D) where {S,T}
  c = cost_model(S, T)[min(D, 3)]
  t = cost_terms(S, D, W, M, J, T)
  length(c) == length(t) || throw(ArgumentError("the cost model of $S has $(length(t)) terms, got $(length(c)) coefficients"))
  # a loop: the broadcast over Int and Float64 terms does not infer for juliac --trim
  x = 0.0
  for i in eachindex(c)
    x += c[i] * t[i]
  end
  x
end

# Beyond the table, W = 17…40, for element types wider than Float64 or an explicit W: the continuous
# Kaiser–Bessel window with the analytic shape parameter β = πW(1 − 1/(2σ)) (Potts & Tasche 2021; at
# W = 12, σ = 2 its aliasing estimate is 4 % above that of the optimised β). The bound est_wc, computed
# as for the table but in BigFloat (256 bits) for W = 12…48 and σ = 1.5, 2, 2.5, is
# exp(r − πW√(1 − 1/σ)) with r between 2.98 and 4.31; `est` takes r = log 80.
const EXTENDED_W = 17:40    # Float128 reaches its rounding floor (≈ 1e-33) at W ≈ 33
const EXTENDED_σ = 1.5:0.25:2.5
function extended_entry(W, σ)
  β = π * W * (1 - 1 / (2σ))
  KernelEntry(KaiserBessel(), W, Float64(σ), β, NaN, 80 * exp(-π * W * sqrt(1 - 1 / σ)),
              amplification(KaiserBessel(), W, σ, β, NaN))
end
const EXTENDED_KERNELS = [extended_entry(W, σ) for σ in EXTENDED_σ for W in EXTENDED_W]

# Bound on the relative error for any nodes: per dimension the aliases of a mode add up to est, the
# tensor-product window to (1 + est)^D − 1. Measured (W = 4…12, D = 1…3): up to 0.51 of it for
# Cartesian nodes, 0.38 for random ones; the RMS estimate est_rms·√D·1.5 is exceeded 2.5× by
# Cartesian nodes.
window_error(e::KernelEntry, D) = expm1(D * log1p(e.est))    # (1 + est)^D − 1 without cancellation

# Rounding floor: grid positions in wide(T) limit the phase to ≈ eps(wide(T))·max(N), the working
# precision T adds ≈ 5·eps(T) (FFT, summation).
error_floor(N, ::Type{T}) where {T} = 0.5 * Float64(eps(wide(T))) * maximum(N) + 5 * Float64(eps(T))

# Rounding error of the FFT beyond the ≈ 5·eps(T) of error_floor: FFTA's FFT loses accuracy with the
# length (measured 3–4400 eps for n = 64…16384 in Float16, Float32, Float64 and Double64, ≤ 0.27·n·eps);
# FFTW and the GPU FFTs stay within a few eps.
fft_error(fft, Ñ, ::Type{T}) where {T} = fft isa FFTABackend ? 0.3 * Float64(eps(T)) * maximum(Ñ) : 0.0

predicted_error(e::KernelEntry, N::NTuple{D,Int}, ::Type{T}, fft=nothing) where {D,T} =
  window_error(e, D) + error_floor(N, T) + 10 * Float64(eps(T)) * e.amp + fft_error(fft, grid_size(e.σ, N, e.W), T)

grid_size(σ, N) = map(n -> smooth_size(ceil(Int, σ * n)), N)
# The grid of a plan has at least W points, so that the halo (W points) wraps around it once.
grid_size(σ, N, W) = map(n -> smooth_size(max(ceil(Int, σ * n), W)), N)

kernel_cost(e::KernelEntry, N::NTuple{D,Int}, J, S=Array, ::Type{T}=Float64) where {D,T} =
  predicted_cost(S, T, e.W, prod(grid_size(e.σ, N)), J, D)

# The kernels a plan may use, from its options `window`, `W`, `σ`, as pairs (entry, predicted
# error): a CustomWindow as given (its error is unknown), otherwise the `n` cheapest table entries
# that reach the tolerance (`choose_kernels`).
function select_kernels(window, W, σ, reltol, N, J, ::Type{T}; Ncost=N, n=1, storage=Array, fft=nothing) where {T}
  check_window_options(W, σ)
  if window isa CustomWindow
    (W === nothing || σ === nothing) && throw(ArgumentError("a CustomWindow needs W and σ"))
    [(KernelEntry(window, W, σ, NaN, NaN, NaN, NaN), NaN)]
  elseif window === :auto || window isa AbstractWindow
    choose_kernels(reltol, N, J, T; window, W, σ, Ncost, n, storage, fft)
  else
    throw(ArgumentError("window must be :auto or an AbstractWindow, got $window"))
  end
end

# σ ≤ 1 gives a grid smaller than the modes, which then alias.
function check_window_options(W, σ)
  W === nothing || (W isa Integer && W >= 2) || throw(ArgumentError("W must be an integer ≥ 2, got $W"))
  σ === nothing || σ > 1 || throw(ArgumentError("σ must be > 1, got $σ"))
  nothing
end

# Indices of the n smallest values, smallest first (n is small: repeated selection).
function cheapest(cost::Vector{Float64}, n)
  idx = Int[]
  for _ in 1:min(n, length(cost))
    i = 0
    for j in eachindex(cost)
      j in idx && continue
      (i == 0 || cost[j] < cost[i]) && (i = j)
    end
    push!(idx, i)
  end
  idx
end

# The `n` table entries of lowest predicted cost that reach `reltol`, restricted by the user's
# `window` (`:auto` or a family), `W` and `σ`, cheapest first, as pairs (entry, predicted error);
# σ is replaced by the user's σ when given. `Ncost` is the size whose oversampled grid the cost
# model sees (NFCT/NFST: the error is that of an NFFT of twice the size on half its grid), `storage`
# selects the cost model, `fft` the FFT backend (its rounding error).
function choose_kernels(reltol, N::NTuple{D,Int}, J, ::Type{T}; window=:auto, W=nothing, σ=nothing, Ncost=N,
                        n=1, storage=Array, fft=nothing) where {D,T}
  cands = kernel_candidates(window, W, σ, T)
  err = [predicted_error(e, N, T, fft) for e in cands]
  emin = minimum(err)
  (W === nothing || σ === nothing) && check_reach(emin, reltol, "N = $N", T, window, W, σ)
  tol = max(reltol, emin)
  ok = cands[err .<= tol]
  # the cost once per entry: it searches the next FFT-friendly grid size. A loop: a closure over
  # `storage` (an abstract `Type`) does not infer, and juliac --trim rejects it.
  cost = zeros(length(ok))
  for (i, e) in enumerate(ok)
    cost[i] = kernel_cost(e, Ncost, J, storage, T)
  end
  best = ok[cheapest(cost, n)]
  σ === nothing ? [(e, predicted_error(e, N, T, fft)) for e in best] :
                  [(with_σ(e, σ), predicted_error(e, N, T, fft)) for e in best]
end

# Table entries for the user's window, W and σ (the nearest tabulated σ); the extension beyond the
# table for an explicit W > 16 or an element type wider than Float64.
function kernel_candidates(window, W, σ, ::Type{T}) where {T}
  extend = (W !== nothing && W in EXTENDED_W) || eps(T) < eps(Float64)
  cands = filter(extend ? vcat(KERNEL_TABLE, EXTENDED_KERNELS) : KERNEL_TABLE) do e
    (window === :auto || e.window == window) && (W === nothing || e.W == W)
  end
  if σ !== nothing && !isempty(cands)
    σtab = argmin(s -> abs(s - σ), unique(e.σ for e in cands))
    cands = filter(e -> e.σ == σtab, cands)
  end
  isempty(cands) && throw(ArgumentError("no kernel for window = $window, W = $W, σ = $σ"))
  cands
end

# Entry e with the user's σ: the extension takes the analytic β for it, a table entry keeps the
# shape parameters of the nearest tabulated σ.
with_σ(e::KernelEntry, σ) = e.W in EXTENDED_W ? extended_entry(e.W, σ) :
  KernelEntry(e.window, e.W, Float64(σ), e.β, e.e0, e.est, e.amp)

# A reltol out of reach throws (only checked unless the user fixed both W and σ): the predicted error
# bounds the measured one, so a plan whose prediction is at most reltol keeps error ≤ reltol, and
# none other is taken.
# repr(window) without `show`, which juliac --trim cannot compile for a Symbol.
window_label(w::Symbol) = ":" * String(w)
window_label(w::AbstractWindow) = String(nameof(typeof(w))) * "()"
window_label(w) = string(w)
function check_reach(reach, reltol, size, ::Type{T}, window, W, σ) where {T}
  reach > reltol || return nothing
  restricted = window === :auto && W === nothing && σ === nothing ? "" : ", window = $(window_label(window)), W = $W, σ = $σ"
  throw(ArgumentError("reltol = $reltol is out of reach for eltype $T, $size$restricted: the most accurate " *
                      "parameters give ≈ $(round(reach; sigdigits=2))"))
end
