# Parameter choice: window, width W, oversampling σ and the window's shape parameters from a
# table of optimised kernels, by predicted error and a fitted cost model.
#
# data/kernel_table.csv: for each window family, W = 4…16 and σ = 1.20:0.05:2.50 the shape
# parameters that minimise the worst-mode aliasing estimate max_ξ √(Σ_{r≠0} φ̂(ξ+r)²)/|φ̂(ξ)|, and
# the RMS of that ratio over the kept modes (est_rms), which predicts the measured relative ℓ²
# error. Computed by the kernel optimiser of the refactoring workspace (bench/micro/kernel_opt.jl).
# The kernel family with a free exponent and the per-(W, σ) optimisation follow ducc (Arras et al.,
# A&A 646, A58, 2021); the table itself is independent of ducc's.

struct KernelEntry
  window::AbstractWindow
  W::Int
  σ::Float64
  β::Float64
  e0::Float64
  est_rms::Float64
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
    KernelEntry(w, W, σ, β, e0, parse(Float64, v[7]), amplification(w, W, σ, β, e0))
  end
end

# Cost model (ms, direct + adjoint NFFT): t ≈ Σᵢ cᵢ·tᵢ with the terms tᵢ of `cost_terms` for the
# storage and coefficients c per dimension (D ≤ 3), storage and element type: the defaults of
# `default_cost`, or those set by `set_cost_model!` (`calibrate_cost`).
#
# CPU: fitted on an M1 Max with 8 threads for J = 2^18, W ∈ 4:16, σ ∈ 1.25:0.25:2.25,
# t ≈ a·J·W^D + a1·J·W + b·Ñ^D·log2(Ñ^D) + c. It only interpolates within that range (median
# deviation 5 % in 2D/3D, 11 % in 1D); the coefficients have no physical meaning.
const COST_COEF = Dict(1 => [2.32e-07, 2.32e-07, 4.38e-07, -1.15],
                       2 => [1.84e-08, 1.33e-06, 1.19e-07, 0.639],
                       3 => [8.17e-08, -2.83e-06, 1.62e-07, 6.17])

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

# Beyond the table, W = 17…48, for element types wider than Float64 or an explicit W: the continuous
# Kaiser–Bessel window with the analytic shape parameter β = πW(1 − 1/(2σ)) (Potts & Tasche 2021; at
# W = 12, σ = 2 its aliasing estimate is 4 % above that of the optimised β). The estimate, computed as
# for the table but in BigFloat (256 bits) for W = 12…48 and σ = 1.5, 2, 2.5, is
# exp(r − πW√(1 − 1/σ)) with r between 0.7 and 2.03; est_rms takes r = log 8, at most 4× too large.
const EXTENDED_W = 17:48
const EXTENDED_σ = 1.5:0.25:2.5
function extended_entry(W, σ)
  β = π * W * (1 - 1 / (2σ))
  KernelEntry(KaiserBessel(), W, Float64(σ), β, NaN, 8 * exp(-π * W * sqrt(1 - 1 / σ)),
              amplification(KaiserBessel(), W, σ, β, NaN))
end
const EXTENDED_KERNELS = [extended_entry(W, σ) for σ in EXTENDED_σ for W in EXTENDED_W]

# Measured relative error ≈ est_rms·√D·(1…1.5); 1.5·√D is about the 90 % quantile over table entries.
window_error(e::KernelEntry, D) = e.est_rms * sqrt(D) * 1.5

# Rounding floor: grid positions in wide(T) limit the phase to ≈ eps(wide(T))·max(N) (position_floor),
# the working precision T adds ≈ 5·eps(T) (FFT, summation).
position_floor(N, ::Type{T}) where {T} = 0.5 * Float64(eps(wide(T))) * maximum(N)
error_floor(N, ::Type{T}) where {T} = position_floor(N, T) + 5 * Float64(eps(T))

predicted_error(e::KernelEntry, N::NTuple{D,Int}, ::Type{T}) where {D,T} =
  window_error(e, D) + error_floor(N, T) + 10 * eps(T) * e.amp

grid_size(σ, N) = map(n -> smooth_size(ceil(Int, σ * n)), N)
# The grid of a plan has at least W points, so that the halo (W points) wraps around it once.
grid_size(σ, N, W) = map(n -> smooth_size(max(ceil(Int, σ * n), W)), N)

kernel_cost(e::KernelEntry, N::NTuple{D,Int}, J, S=Array, ::Type{T}=Float64) where {D,T} =
  predicted_cost(S, T, e.W, prod(grid_size(e.σ, N)), J, D)

# The kernels a plan may use, from its options `window`, `W`, `σ`, as pairs (entry, predicted
# error): a CustomWindow as given (its error is unknown), otherwise the `n` cheapest table entries
# that reach the tolerance (`choose_kernels`).
function select_kernels(window, W, σ, reltol, N, J, ::Type{T}; Ncost=N, n=1, storage=Array) where {T}
  check_window_options(W, σ)
  if window isa CustomWindow
    (W === nothing || σ === nothing) && throw(ArgumentError("a CustomWindow needs W and σ"))
    [(KernelEntry(window, W, σ, NaN, NaN, NaN, NaN), NaN)]
  elseif window === :auto || window isa AbstractWindow
    choose_kernels(reltol, N, J, T; window, W, σ, Ncost, n, storage)
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
# selects the cost model.
function choose_kernels(reltol, N::NTuple{D,Int}, J, ::Type{T}; window=:auto, W=nothing, σ=nothing, Ncost=N,
                        n=1, storage=Array) where {D,T}
  cands = kernel_candidates(window, W, σ, T)
  err = [predicted_error(e, N, T) for e in cands]
  emin = minimum(err)
  (W === nothing || σ === nothing) && check_reach(emin - position_floor(N, T), reltol, N, T, window, W, σ)
  tol = max(reltol, emin)
  ok = cands[err .<= tol]
  # the cost once per entry: it searches the next FFT-friendly grid size. A loop: a closure over
  # `storage` (an abstract `Type`) does not infer, and juliac --trim rejects it.
  cost = zeros(length(ok))
  for (i, e) in enumerate(ok)
    cost[i] = kernel_cost(e, Ncost, J, storage, T)
  end
  best = ok[cheapest(cost, n)]
  σ === nothing ? [(e, predicted_error(e, N, T)) for e in best] :
                  [(with_σ(e, σ), predicted_error(e, N, T)) for e in best]
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
  KernelEntry(e.window, e.W, Float64(σ), e.β, e.e0, e.est_rms, e.amp)

# A reltol out of reach throws (only checked unless the user fixed both W and σ). The predicted
# error is conservative (measured errors 3–10× below), so only a predicted error above 10·reltol
# counts as out of reach; up to that, the most accurate candidate is taken. `reach` leaves out the
# rounding floor of the node positions: it depends on N, not on the parameters, and the interface
# contract allows it (≈ eps·max(N)).
function check_reach(reach, reltol, N, ::Type{T}, window, W, σ) where {T}
  reach > 10 * reltol || return nothing
  restricted = window === :auto && W === nothing && σ === nothing ? "" : ", window = $(repr(window)), W = $W, σ = $σ"
  throw(ArgumentError("reltol = $reltol is out of reach for eltype $T, N = $N$restricted: the most accurate " *
                      "parameters give ≈ $(round(reach; sigdigits=2))"))
end
