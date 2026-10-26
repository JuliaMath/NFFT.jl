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

# The resampling correction multiplies the modes at the edge of the band by up to A = φ̂(0)/φ̂(1/(2σ)),
# and with them the rounding errors of the grid values: the error grows by ≈ 10·eps(T)·A (Float32,
# W = 10, σ = 1.2: A = 140, measured 1.6e-4 for nodes on grid points). Large for small σ and large W.
function amplification(window, W, σ, β, e0)
  φ = window_function(window, W, β, e0)
  x, w = gausslegendre(2W + 20)
  z = x .* (W / 2); wz = w .* (W / 2) .* φ.(z)
  φ̂(ξ) = sum(wz[q] * cospi(2ξ * z[q]) for q in eachindex(z))
  φ̂(0.0) / φ̂(1 / (2σ))
end

const KERNEL_TABLE = let lines = readlines(joinpath(@__DIR__, "..", "data", "kernel_table.csv"))[2:end]
  map(split.(lines, ",")) do v
    w = table_window(v[1]); W = parse(Int, v[2]); σ = parse(Float64, v[3])
    β = parse(Float64, v[4]); e0 = parse(Float64, v[5])
    KernelEntry(w, W, σ, β, e0, parse(Float64, v[7]), amplification(w, W, σ, β, e0))
  end
end

# Cost model (ms, direct + adjoint NFFT), fitted on an M1 Max with 8 threads for J = 2^18,
# W ∈ 4:16, σ ∈ 1.25:0.25:2.25:  t ≈ a·J·W^D + a1·J·W + b·Ñ^D·log2(Ñ^D) + c. It only interpolates
# within that range (median deviation 5 % in 2D/3D, 11 % in 1D); the coefficients have no
# physical meaning.
const COST_COEF = Dict(1 => (2.32e-07, 2.32e-07, 4.38e-07, -1.15),
                       2 => (1.84e-08, 1.33e-06, 1.19e-07, 0.639),
                       3 => (8.17e-08, -2.83e-06, 1.62e-07, 6.17))

# Measured relative error ≈ est_rms·√D·(1…1.5); 1.5·√D is about the 90 % quantile over table entries.
window_error(e::KernelEntry, D) = e.est_rms * sqrt(D) * 1.5

# Rounding floor: node coordinates in Float64 limit the phase to ≈ eps(Float64)·max(N), the
# working precision T adds ≈ 5·eps(T) (FFT, summation).
error_floor(N, T) = 0.5 * eps(Float64) * maximum(N) + 5 * eps(T)

predicted_error(e::KernelEntry, N::NTuple{D,Int}, T) where {D} =
  window_error(e, D) + error_floor(N, T) + 10 * eps(T) * e.amp

grid_size(σ, N) = map(n -> smooth_size(ceil(Int, σ * n)), N)

function kernel_cost(e::KernelEntry, N::NTuple{D,Int}, J) where {D}
  a, a1, b, c = COST_COEF[min(D, 3)]
  M = prod(grid_size(e.σ, N))
  a * J * e.W^D + a1 * J * e.W + b * M * log2(M) + c
end

# Choose the kernel for requested `reltol` from the table, restricted by the user's `window`
# (`:auto` or a family), `W` and `σ`. Returns the table entry (with σ replaced by the user's σ when
# given) and the predicted error.
function choose_kernel(reltol, N::NTuple{D,Int}, J, T; window=:auto, W=nothing, σ=nothing) where {D}
  cands = filter(KERNEL_TABLE) do e
    (window === :auto || e.window == window) && (W === nothing || e.W == W)
  end
  if σ !== nothing
    isempty(cands) || (σtab = argmin(s -> abs(s - σ), unique(e.σ for e in cands));
                       cands = filter(e -> e.σ == σtab, cands))
  end
  isempty(cands) && throw(ArgumentError("no table entry for window = $window, W = $W, σ = $σ"))
  # below what any candidate reaches: take the most accurate
  tol = max(reltol, minimum(e -> predicted_error(e, N, T), cands))
  best = argmin(e -> kernel_cost(e, N, J), filter(e -> predicted_error(e, N, T) <= tol, cands))
  if σ !== nothing
    best = KernelEntry(best.window, best.W, Float64(σ), best.β, best.e0, best.est_rms, best.amp)
  end
  best, predicted_error(best, N, T)
end
